//! Native MLX Gemma 4 vision tower.
//!
//! Gemma 4's image encoder is **not** the SigLIP tower Gemma 3 used. It is a
//! native-resolution ViT (`model_type: "gemma4_vision"`) with:
//!   * linear patch embedding over 16×16 RGB patches (no conv), plus a
//!     factorized 2-D absolute position table (`[2, 10240, 1152]`, x + y),
//!   * 27 Gemma-shaped blocks (RMSNorm pre/post around both attn and MLP,
//!     QK-norm, GeGLU) with **2-D RoPE** and **bidirectional** attention,
//!   * `attn scale = 1.0` (the q_norm absorbs the 1/sqrt(head_dim)),
//!   * 3×3 average pooling → ×sqrt(hidden) → standardize (`std_bias`/`std_scale`),
//!   * `embed_vision`: RMSNorm-without-scale → quantized 1152→2816 projection.
//!
//! Two conventions here differ from the text tower and are easy to get wrong,
//! so they are called out at the point of use below: the vision RMSNorm is a
//! plain `normed * weight` (**not** the text `normed * (1 + weight)`), and the
//! attention scale is 1.0.
//!
//! Reference: `transformers/models/gemma4/modeling_gemma4.py` (main).
//! Numerically validated against `scripts/gemma4_vision_ref.py` golden tensors
//! by `tests/gemma4_vision_parity.rs`.

#[cfg(feature = "mlx-native")]
pub mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use mlx_rs::Dtype;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};

    use std::sync::OnceLock;

    use crate::native_quant::quantized_matmul_with_mode;

    // ───────────────────────────── config ─────────────────────────────

    // The `vision_config` block is pure serde + arithmetic validation, so it
    // lives in the ungated `gemma4_config` module alongside the rest of this
    // config.json. Re-exported here so call sites are untouched.
    pub use crate::gemma4_config::{NativeGemma4VisionConfig, NativeGemma4VisionRope};

    /// Run the tower in float32 instead of the checkpoint's bf16.
    /// Used by the golden-tensor parity test, which compares against a
    /// float32 PyTorch reference; production leaves this off.
    fn vision_f32_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("LUMEN_VISION_F32")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    /// Drain the lazy graph every N encoder layers. `0` disables draining
    /// (fastest, highest peak memory). Default 4 keeps the live set to a few
    /// layers' activations, which matters on 36 GB machines where the 26B text
    /// weights already occupy ~15 GB.
    fn vision_eval_every() -> usize {
        static EVERY: OnceLock<usize> = OnceLock::new();
        *EVERY.get_or_init(|| {
            std::env::var("LUMEN_VISION_EVAL_EVERY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
        })
    }

    /// Per-image soft-token budget. Lower values shrink the patch grid
    /// quadratically in attention cost and linearly in activation memory
    /// (280 → 2520 patches max, 140 → 1260, 70 → 630). Must be one of
    /// [`super::SUPPORTED_SOFT_TOKENS`]; anything else falls back to the
    /// config value.
    pub fn soft_token_budget_override() -> Option<usize> {
        static BUDGET: OnceLock<Option<usize>> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            let raw = std::env::var("LUMEN_VISION_MAX_SOFT_TOKENS").ok()?;
            let parsed: usize = raw.parse().ok()?;
            if super::SUPPORTED_SOFT_TOKENS.contains(&parsed) {
                Some(parsed)
            } else {
                eprintln!(
                    "[vision] ignoring LUMEN_VISION_MAX_SOFT_TOKENS={raw} \
                     (must be one of {:?})",
                    super::SUPPORTED_SOFT_TOKENS
                );
                None
            }
        })
    }

    // ───────────────────────────── weights ─────────────────────────────

    /// One encoder block. All projections are stored **pre-transposed**
    /// (`[in, out]`) so the forward path is a plain `matmul` with no per-call
    /// transpose — the tower runs 27× per image and transposes are not free.
    struct VisionLayer {
        input_layernorm: Array,
        post_attention_layernorm: Array,
        pre_feedforward_layernorm: Array,
        post_feedforward_layernorm: Array,
        q_proj_t: Array,
        k_proj_t: Array,
        v_proj_t: Array,
        o_proj_t: Array,
        q_norm: Array,
        k_norm: Array,
        gate_proj_t: Array,
        up_proj_t: Array,
        down_proj_t: Array,
    }

    /// Quantization params for `embed_vision.embedding_projection`, resolved
    /// from the checkpoint's `quantization` block by
    /// [`crate::gemma4::quant_params_for`].
    ///
    /// Shipped checkpoints disagree here — `mlx-community/gemma-4-26b-a4b-it-4bit`
    /// is affine 4-bit group 64 with biases, `…-it-nvfp4` is nvfp4 4-bit group
    /// 16 without them — and the packed tensors look identical, so this has to
    /// be carried in rather than sniffed.
    #[derive(Debug, Clone, Copy)]
    pub struct VisionProjectionQuant {
        pub group_size: i32,
        pub bits: i32,
        pub mode: &'static std::ffi::CStr,
    }

    /// Quantized `embed_vision.embedding_projection`.
    struct VisionProjection {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: &'static std::ffi::CStr,
    }

    pub struct NativeGemma4VisionTower {
        cfg: NativeGemma4VisionConfig,
        /// `[3·patch², hidden]`, pre-transposed.
        input_proj_t: Array,
        /// x/y halves of `position_embedding_table`, each `[position_embedding_size, hidden]`.
        pos_table_x: Array,
        pos_table_y: Array,
        std_bias: Array,
        std_scale: Array,
        /// `ones([head_dim])` for the scale-less v_norm — allocated once.
        ones_head_dim: Array,
        layers: Vec<VisionLayer>,
        projection: VisionProjection,
        /// Working dtype: bf16 in production, f32 under `LUMEN_VISION_F32`.
        dtype: Dtype,
    }

    /// Look up a weight and cast it to the working dtype.
    fn take(
        weights: &std::collections::HashMap<String, Array>,
        name: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        weights
            .get(name)
            .ok_or_else(|| anyhow!("vision weight `{name}` not found"))?
            .as_dtype(dtype)
            .with_context(|| format!("cast `{name}` to {dtype:?}"))
    }

    /// Look up a `[out, in]` projection and return it transposed to `[in, out]`.
    fn take_t(
        weights: &std::collections::HashMap<String, Array>,
        name: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        let w = take(weights, name, dtype)?;
        mlx_rs::ops::transpose(&w).with_context(|| format!("transpose(`{name}`)"))
    }

    impl NativeGemma4VisionTower {
        /// Build the tower from the raw (un-sanitized) weight bag.
        ///
        /// Call this **before** `NativeGemma4Weights::sanitize()` strips the
        /// `vision_tower.*` / `embed_vision.*` entries, or from a bag loaded
        /// with vision retention enabled.
        pub fn load(
            weights: &std::collections::HashMap<String, Array>,
            cfg: NativeGemma4VisionConfig,
            quant: VisionProjectionQuant,
        ) -> Result<Self> {
            cfg.validate()?;
            let dtype = if vision_f32_enabled() {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            };

            let pos_table = weights
                .get("vision_tower.patch_embedder.position_embedding_table")
                .ok_or_else(|| anyhow!("vision weight `position_embedding_table` not found"))?;
            if pos_table.ndim() != 3 || pos_table.shape()[0] != 2 {
                return Err(anyhow!(
                    "position_embedding_table must be [2, P, hidden], got {:?}",
                    pos_table.shape()
                ));
            }
            let pos_table = pos_table
                .as_dtype(dtype)
                .context("cast position_embedding_table")?;
            let pos_table_x = pos_table.index(0);
            let pos_table_y = pos_table.index(1);

            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for i in 0..cfg.num_hidden_layers {
                let p = format!("vision_tower.encoder.layers.{i}.");
                layers.push(VisionLayer {
                    input_layernorm: take(weights, &format!("{p}input_layernorm.weight"), dtype)?,
                    post_attention_layernorm: take(
                        weights,
                        &format!("{p}post_attention_layernorm.weight"),
                        dtype,
                    )?,
                    pre_feedforward_layernorm: take(
                        weights,
                        &format!("{p}pre_feedforward_layernorm.weight"),
                        dtype,
                    )?,
                    post_feedforward_layernorm: take(
                        weights,
                        &format!("{p}post_feedforward_layernorm.weight"),
                        dtype,
                    )?,
                    q_proj_t: take_t(
                        weights,
                        &format!("{p}self_attn.q_proj.linear.weight"),
                        dtype,
                    )?,
                    k_proj_t: take_t(
                        weights,
                        &format!("{p}self_attn.k_proj.linear.weight"),
                        dtype,
                    )?,
                    v_proj_t: take_t(
                        weights,
                        &format!("{p}self_attn.v_proj.linear.weight"),
                        dtype,
                    )?,
                    o_proj_t: take_t(
                        weights,
                        &format!("{p}self_attn.o_proj.linear.weight"),
                        dtype,
                    )?,
                    q_norm: take(weights, &format!("{p}self_attn.q_norm.weight"), dtype)?,
                    k_norm: take(weights, &format!("{p}self_attn.k_norm.weight"), dtype)?,
                    gate_proj_t: take_t(
                        weights,
                        &format!("{p}mlp.gate_proj.linear.weight"),
                        dtype,
                    )?,
                    up_proj_t: take_t(weights, &format!("{p}mlp.up_proj.linear.weight"), dtype)?,
                    down_proj_t: take_t(
                        weights,
                        &format!("{p}mlp.down_proj.linear.weight"),
                        dtype,
                    )?,
                });
            }

            // The projection stays quantized — dequantizing 2816×1152 per image
            // would cost more than the qmm itself.
            let proj_w = weights
                .get("embed_vision.embedding_projection.weight")
                .ok_or_else(|| anyhow!("embed_vision.embedding_projection.weight not found"))?
                .clone();
            // Scales and biases are taken **as stored**, never cast. Their
            // dtype is part of the quantization format, not a working
            // precision: affine keeps bf16 scales, but nvfp4 packs E4M3 into
            // u8, and casting those to bf16 reinterprets the format and makes
            // the FFI reject the call. The text path's `require_clone` does the
            // same thing for exactly this reason.
            let proj_s = weights
                .get("embed_vision.embedding_projection.scales")
                .ok_or_else(|| anyhow!("embed_vision.embedding_projection.scales not found"))?
                .clone();
            // Absent under mxfp4 / nvfp4, which carry no zero-point.
            let proj_b = weights
                .get("embed_vision.embedding_projection.biases")
                .cloned();

            // `(group_size, bits, mode)` come from the checkpoint's own
            // quantization block, not from the tensor shapes. Shapes cannot
            // distinguish affine-4/64 from nvfp4-4/16 packed the same way, and
            // guessing "affine" for an nvfp4 checkpoint dequantizes to garbage
            // with no error anywhere — the tower would just emit confident
            // nonsense. The shape check below is a cross-check on that config,
            // not the source of truth.
            let (group_size, bits, mode) = (quant.group_size, quant.bits, quant.mode);
            let in_features = cfg.hidden_size as i32;
            let n_groups = proj_s.shape()[1];
            if n_groups <= 0 || group_size <= 0 || n_groups * group_size != in_features {
                return Err(anyhow!(
                    "projection scales {:?} imply {} groups, which does not tile hidden_size {} \
                     at the configured group_size {}",
                    proj_s.shape(),
                    n_groups,
                    in_features,
                    group_size
                ));
            }
            let per_word = 32 / bits;
            if per_word <= 0 || proj_w.shape()[1] * per_word != in_features {
                return Err(anyhow!(
                    "projection weight {:?} does not hold {} × {}-bit values per row",
                    proj_w.shape(),
                    in_features,
                    bits
                ));
            }

            let ones_head_dim = mlx_rs::ops::ones::<f32>(&[cfg.head_dim as i32])
                .context("alloc ones(head_dim)")?
                .as_dtype(dtype)
                .context("cast ones(head_dim)")?;

            Ok(Self {
                input_proj_t: take_t(
                    weights,
                    "vision_tower.patch_embedder.input_proj.weight",
                    dtype,
                )?,
                pos_table_x,
                pos_table_y,
                std_bias: take(weights, "vision_tower.std_bias", Dtype::Float32)?,
                std_scale: take(weights, "vision_tower.std_scale", Dtype::Float32)?,
                ones_head_dim,
                layers,
                projection: VisionProjection {
                    weight: proj_w,
                    scales: proj_s,
                    biases: proj_b,
                    group_size,
                    bits,
                    mode,
                },
                cfg,
                dtype,
            })
        }

        pub fn config(&self) -> &NativeGemma4VisionConfig {
            &self.cfg
        }

        /// Encode one image into soft tokens for the language model.
        ///
        /// * `pixel_values` — `[num_patches, 3·patch²]`, already rescaled to
        ///   `[0, 1]` and patchified in raster order (see [`super::preprocess`]).
        /// * `grid` — `(patch_rows, patch_cols)`; both must be multiples of
        ///   `pooling_kernel_size` and their product must equal `num_patches`.
        ///
        /// Returns `[num_patches / pooling_kernel_size², text_hidden_size]`.
        ///
        /// Padding patches are deliberately not supported: upstream pads only
        /// to batch differently-sized images, and `scripts/gemma4_vision_ref.py`
        /// asserts a single unpadded image yields identical soft tokens. That
        /// lets this path skip the attention mask entirely.
        pub fn forward(&self, pixel_values: &Array, grid: (usize, usize)) -> Result<Array> {
            let (rows, cols) = grid;
            let k = self.cfg.pooling_kernel_size;
            let n = rows * cols;

            if pixel_values.ndim() != 2 {
                return Err(anyhow!(
                    "pixel_values must be [num_patches, 3·patch²], got {:?}",
                    pixel_values.shape()
                ));
            }
            if pixel_values.shape()[0] as usize != n {
                return Err(anyhow!(
                    "pixel_values has {} patches but grid {rows}×{cols} implies {n}",
                    pixel_values.shape()[0]
                ));
            }
            if rows % k != 0 || cols % k != 0 {
                return Err(anyhow!(
                    "patch grid {rows}×{cols} is not divisible by pooling kernel {k}"
                ));
            }
            let expect_in = 3 * self.cfg.patch_size * self.cfg.patch_size;
            if pixel_values.shape()[1] as usize != expect_in {
                return Err(anyhow!(
                    "pixel_values width {} != 3·patch² ({expect_in})",
                    pixel_values.shape()[1]
                ));
            }

            let hidden = self.cfg.hidden_size as i32;
            let heads = self.cfg.num_attention_heads as i32;
            let head_dim = self.cfg.head_dim as i32;
            let eps = self.cfg.rms_norm_eps;
            let n_i32 = n as i32;

            // ── patch embedder ──
            // Gemma 4 applies no dataset normalization; it rescales in model
            // code instead: [0,1] → [-1,1].
            let px = pixel_values
                .as_dtype(self.dtype)
                .context("cast pixel_values")?;
            let half = Array::from_f32(0.5).as_dtype(self.dtype)?;
            let two = Array::from_f32(2.0).as_dtype(self.dtype)?;
            let px = mlx_rs::ops::multiply(&mlx_rs::ops::subtract(&px, &half)?, &two)
                .context("patch_embed: 2·(x-0.5)")?;
            let mut h = mlx_rs::ops::matmul(&px, &self.input_proj_t)
                .context("patch_embed: input_proj matmul")?;

            let (x_ids, y_ids) = raster_position_ids(rows, cols)?;
            let pos_x = self
                .pos_table_x
                .take_axis(&x_ids, 0)
                .context("patch_embed: position table x lookup")?;
            let pos_y = self
                .pos_table_y
                .take_axis(&y_ids, 0)
                .context("patch_embed: position table y lookup")?;
            h = mlx_rs::ops::add(&h, &mlx_rs::ops::add(&pos_x, &pos_y)?)
                .context("patch_embed: + position embedding")?;
            // [N, C] → [1, N, C]
            let mut h = mlx_rs::ops::reshape(&h, &[1, n_i32, hidden])
                .context("patch_embed: reshape to [1, N, C]")?;

            // ── 2-D RoPE tables (position-dependent only: built once, reused
            //    by all 27 layers) ──
            let (cos, sin) = self.build_2d_rope(&x_ids, &y_ids, n_i32)?;

            // ── encoder ──
            //
            // Drain the lazy graph every `eval_every` layers. Without this MLX
            // keeps every layer's intermediates alive until the first eval:
            // at 2394 patches each layer holds ~60 MB of MLP activations
            // ([N, intermediate] bf16 × gate/up/activated), so a 27-layer run
            // peaks multiple GB above what it needs. The text decoder does the
            // same thing via `LUMEN_GEMMA4_ASYNC_EVAL_EVERY_K`.
            let eval_every = vision_eval_every();
            for (idx, lw) in self.layers.iter().enumerate() {
                h = self
                    .encoder_layer(&h, lw, &cos, &sin, n_i32, heads, head_dim, eps)
                    .with_context(|| format!("vision encoder layer {idx}"))?;
                if eval_every > 0 && (idx + 1) % eval_every == 0 {
                    h.eval().context("vision encoder: periodic eval")?;
                }
            }

            // ── pooler ──
            // Upstream builds a one-hot matrix from patch positions; with a
            // regular unpadded grid that is exactly a 3×3 mean over the raster
            // layout, so reshape+mean gives the same result far more cheaply.
            let k_i32 = k as i32;
            let pooled = mlx_rs::ops::reshape(
                &h,
                &[(rows / k) as i32, k_i32, (cols / k) as i32, k_i32, hidden],
            )
            .context("pooler: reshape to [R,k,C,k,H]")?;
            let pooled = mlx_rs::ops::mean_axes(&pooled, &[1, 3], false)
                .context("pooler: mean over 3×3 kernel")?;
            let soft_len = ((rows / k) * (cols / k)) as i32;
            let pooled = mlx_rs::ops::reshape(&pooled, &[soft_len, hidden])
                .context("pooler: reshape to [S, H]")?;

            // ×sqrt(hidden) then standardize. Upstream keeps this in float32
            // because the scaling can overflow float16; we follow suit.
            let pooled = pooled
                .as_dtype(Dtype::Float32)
                .context("pooler: cast to f32")?;
            let root = Array::from_f32((self.cfg.hidden_size as f32).sqrt());
            let pooled = mlx_rs::ops::multiply(&pooled, &root).context("pooler: ×sqrt(hidden)")?;
            let pooled = if self.cfg.standardize {
                let centered =
                    mlx_rs::ops::subtract(&pooled, &self.std_bias).context("pooler: -std_bias")?;
                mlx_rs::ops::multiply(&centered, &self.std_scale).context("pooler: ×std_scale")?
            } else {
                pooled
            };

            // ── embed_vision: RMSNorm(no scale) → quantized 1152→2816 ──
            let pooled = pooled.as_dtype(self.dtype).context("projection: cast")?;
            let normed =
                crate::native_norm::rms_norm(&pooled, &self.ones_head_dim_like_hidden()?, eps)
                    .context("projection: pre-projection RMSNorm")?;
            quantized_matmul_with_mode(
                &normed,
                &self.projection.weight,
                &self.projection.scales,
                self.projection.biases.as_ref(),
                /* transpose */ true,
                self.projection.group_size,
                self.projection.bits,
                self.projection.mode,
            )
            .context("projection: quantized_matmul")
        }

        /// `ones([hidden_size])` for the scale-less pre-projection RMSNorm.
        fn ones_head_dim_like_hidden(&self) -> Result<Array> {
            mlx_rs::ops::ones::<f32>(&[self.cfg.hidden_size as i32])
                .context("alloc ones(hidden)")?
                .as_dtype(self.dtype)
                .context("cast ones(hidden)")
        }

        /// Build the 2-D RoPE cos/sin tables, shape `[N, head_dim]` each.
        ///
        /// Frequencies are computed **per spatial dim** over `head_dim/2`
        /// channels (so x and y share an identical frequency range), then the
        /// two dims' tables are concatenated along the last axis.
        fn build_2d_rope(&self, x_ids: &Array, y_ids: &Array, n: i32) -> Result<(Array, Array)> {
            let head_dim = self.cfg.head_dim as i32;
            let spatial = head_dim / 2;
            let theta = self.cfg.rope_parameters.rope_theta;

            let half = (spatial / 2) as usize;
            let mut inv: Vec<f32> = Vec::with_capacity(half);
            for i in 0..half {
                let exp = (2 * i) as f32 / spatial as f32;
                inv.push(1.0 / theta.powf(exp));
            }
            let inv_freq = Array::from_slice(&inv, &[1, half as i32]);

            let mut coses = Vec::with_capacity(2);
            let mut sins = Vec::with_capacity(2);
            for ids in [x_ids, y_ids] {
                let pos = ids
                    .as_dtype(Dtype::Float32)
                    .context("rope: cast positions")?;
                let pos = mlx_rs::ops::reshape(&pos, &[n, 1]).context("rope: reshape positions")?;
                let freqs =
                    mlx_rs::ops::multiply(&pos, &inv_freq).context("rope: pos × inv_freq")?;
                let emb = mlx_rs::ops::concatenate_axis(&[&freqs, &freqs], -1)
                    .context("rope: concat(freqs, freqs)")?;
                coses.push(mlx_rs::ops::cos(&emb).context("rope: cos")?);
                sins.push(mlx_rs::ops::sin(&emb).context("rope: sin")?);
            }
            let cos = mlx_rs::ops::concatenate_axis(&[&coses[0], &coses[1]], -1)
                .context("rope: concat cos over dims")?
                .as_dtype(self.dtype)?;
            let sin = mlx_rs::ops::concatenate_axis(&[&sins[0], &sins[1]], -1)
                .context("rope: concat sin over dims")?
                .as_dtype(self.dtype)?;
            Ok((cos, sin))
        }

        /// Apply 2-D RoPE to `x` of shape `[1, N, H, head_dim]`.
        ///
        /// The head_dim is split into `ndim = 2` contiguous chunks; each chunk
        /// is rotated with its own spatial dim's cos/sin.
        fn apply_2d_rope(&self, x: &Array, cos: &Array, sin: &Array, n: i32) -> Result<Array> {
            let head_dim = self.cfg.head_dim as i32;
            let per = 2 * (head_dim / 4); // = head_dim / 2 for even head_dim
            let mut parts = Vec::with_capacity(2);
            for k in 0..2 {
                let lo = k * per;
                let hi = lo + per;
                let xs = x.index((Ellipsis, lo..hi));
                // [N, per] → [1, N, 1, per] so it broadcasts across heads.
                let c = mlx_rs::ops::reshape(cos.index((Ellipsis, lo..hi)), &[1, n, 1, per])
                    .context("rope: reshape cos slice")?;
                let s = mlx_rs::ops::reshape(sin.index((Ellipsis, lo..hi)), &[1, n, 1, per])
                    .context("rope: reshape sin slice")?;
                let rotated = rotate_half(&xs, per)?;
                let a = mlx_rs::ops::multiply(&xs, &c).context("rope: x·cos")?;
                let b = mlx_rs::ops::multiply(&rotated, &s).context("rope: rotate_half(x)·sin")?;
                parts.push(mlx_rs::ops::add(&a, &b).context("rope: x·cos + rot·sin")?);
            }
            mlx_rs::ops::concatenate_axis(&[&parts[0], &parts[1]], -1)
                .context("rope: concat rotated chunks")
        }

        #[allow(clippy::too_many_arguments)]
        fn encoder_layer(
            &self,
            h: &Array,
            lw: &VisionLayer,
            cos: &Array,
            sin: &Array,
            n: i32,
            heads: i32,
            head_dim: i32,
            eps: f32,
        ) -> Result<Array> {
            let hidden = self.cfg.hidden_size as i32;
            let residual = h.clone();

            // NOTE: vision RMSNorm is `normed * weight` — mlx fast::rms_norm's
            // exact semantics. The *text* tower's Gemma convention (1 + weight)
            // does not apply here.
            let x = crate::native_norm::rms_norm(h, &lw.input_layernorm, eps)
                .context("input_layernorm")?;

            let q = mlx_rs::ops::matmul(&x, &lw.q_proj_t).context("q_proj")?;
            let k = mlx_rs::ops::matmul(&x, &lw.k_proj_t).context("k_proj")?;
            let v = mlx_rs::ops::matmul(&x, &lw.v_proj_t).context("v_proj")?;

            let q = mlx_rs::ops::reshape(&q, &[1, n, heads, head_dim]).context("reshape Q")?;
            let k = mlx_rs::ops::reshape(&k, &[1, n, heads, head_dim]).context("reshape K")?;
            let v = mlx_rs::ops::reshape(&v, &[1, n, heads, head_dim]).context("reshape V")?;

            let q = crate::native_norm::rms_norm(&q, &lw.q_norm, eps).context("q_norm")?;
            let k = crate::native_norm::rms_norm(&k, &lw.k_norm, eps).context("k_norm")?;
            // v_norm has no learnable scale (`with_scale=False` upstream).
            let v = crate::native_norm::rms_norm(&v, &self.ones_head_dim, eps).context("v_norm")?;

            let q = self.apply_2d_rope(&q, cos, sin, n)?;
            let k = self.apply_2d_rope(&k, cos, sin, n)?;

            let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3]).context("transpose Q")?;
            let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3]).context("transpose K")?;
            let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3]).context("transpose V")?;

            // scale = 1.0, not 1/sqrt(head_dim): Gemma4VisionAttention sets
            // `self.scaling = 1.0` and lets q_norm absorb the factor.
            // Bidirectional (no mask) — every patch sees every patch, and the
            // unpadded contract means there is nothing to mask out.
            let attn = crate::native_attention::sdpa(&q, &k, &v, 1.0, false).context("sdpa")?;

            let attn =
                mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3]).context("transpose attn back")?;
            let attn = mlx_rs::ops::reshape(&attn, &[1, n, heads * head_dim])
                .context("reshape attn flat")?;
            let attn = mlx_rs::ops::matmul(&attn, &lw.o_proj_t).context("o_proj")?;

            let attn = crate::native_norm::rms_norm(&attn, &lw.post_attention_layernorm, eps)
                .context("post_attention_layernorm")?;
            let h = mlx_rs::ops::add(&residual, &attn).context("+residual (attn)")?;

            let residual = h.clone();
            let x = crate::native_norm::rms_norm(&h, &lw.pre_feedforward_layernorm, eps)
                .context("pre_feedforward_layernorm")?;
            let gate = mlx_rs::ops::matmul(&x, &lw.gate_proj_t).context("gate_proj")?;
            let up = mlx_rs::ops::matmul(&x, &lw.up_proj_t).context("up_proj")?;
            let activated = geglu(&gate, &up)?;
            let down = mlx_rs::ops::matmul(&activated, &lw.down_proj_t).context("down_proj")?;
            let down = crate::native_norm::rms_norm(&down, &lw.post_feedforward_layernorm, eps)
                .context("post_feedforward_layernorm")?;
            let out = mlx_rs::ops::add(&residual, &down).context("+residual (mlp)")?;
            debug_assert_eq!(out.shape(), [1, n, hidden]);
            Ok(out)
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    /// `(x_ids, y_ids)` for a `rows × cols` patch grid in raster order,
    /// matching upstream's `meshgrid(arange(cols), arange(rows), indexing="xy")`
    /// followed by `reshape(N, 2)`: index `n = y·cols + x`.
    fn raster_position_ids(rows: usize, cols: usize) -> Result<(Array, Array)> {
        let n = rows * cols;
        let mut xs: Vec<i32> = Vec::with_capacity(n);
        let mut ys: Vec<i32> = Vec::with_capacity(n);
        for y in 0..rows {
            for x in 0..cols {
                xs.push(x as i32);
                ys.push(y as i32);
            }
        }
        Ok((
            Array::from_slice(&xs, &[n as i32]),
            Array::from_slice(&ys, &[n as i32]),
        ))
    }

    /// `cat(-x[..., d/2:], x[..., :d/2])` along the last axis.
    fn rotate_half(x: &Array, d: i32) -> Result<Array> {
        let h = d / 2;
        let first = x.index((Ellipsis, 0..h));
        let second = x.index((Ellipsis, h..d));
        let neg = mlx_rs::ops::negative(&second).context("rotate_half: negate")?;
        mlx_rs::ops::concatenate_axis(&[&neg, &first], -1).context("rotate_half: concat")
    }

    /// GeGLU with the tanh gelu approximation (`gelu_pytorch_tanh`).
    fn geglu(gate: &Array, up: &Array) -> Result<Array> {
        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.797_884_6_f32).as_dtype(dt)?;
        let x_sq = mlx_rs::ops::multiply(gate, gate)?;
        let x_cubed = mlx_rs::ops::multiply(&x_sq, gate)?;
        let inner = mlx_rs::ops::add(gate, &mlx_rs::ops::multiply(&x_cubed, &c3)?)?;
        let t = mlx_rs::ops::tanh(&mlx_rs::ops::multiply(&coeff, &inner)?)?;
        let gelu = mlx_rs::ops::multiply(
            &mlx_rs::ops::multiply(&half, gate)?,
            &mlx_rs::ops::add(&one, &t)?,
        )?;
        mlx_rs::ops::multiply(&gelu, up).context("geglu: gelu(gate) × up")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{NativeGemma4VisionConfig, NativeGemma4VisionTower, VisionProjectionQuant};

// ─────────────────────────── image preprocessing ───────────────────────────
//
// Pure CPU, no MLX — mirrors `Gemma4ImageProcessor._preprocess`.

use crate::vision_image::{decode_bounded, dimensions_bounded};

/// Soft-token budgets the upstream processor accepts.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// How many `image_token_id` placeholders this image will need, without
/// decoding a single pixel.
///
/// The count is a pure function of the image's dimensions — they fix the
/// resized patch grid, and the grid fixes the pooled soft-token count — so the
/// header is enough. That lets the server size a prompt (context guard, usage
/// accounting) for a request it may end up rejecting, without paying for a
/// decode it would then throw away.
pub fn soft_token_count(
    encoded: &[u8],
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<usize, String> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(format!(
            "max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {max_soft_tokens}"
        ));
    }
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (w, h) = dimensions_bounded(encoded)?;
    let (target_h, target_w) = aspect_ratio_preserving_size(
        h as usize,
        w as usize,
        patch_size,
        max_patches,
        pooling_kernel_size,
    )?;
    Ok((target_h / patch_size / pooling_kernel_size)
        * (target_w / patch_size / pooling_kernel_size))
}

/// Result of preparing one image for the vision tower.
pub struct PreparedImage {
    /// `[num_patches × (3·patch²)]` in raster order, rescaled to `[0, 1]`.
    pub pixel_values: Vec<f32>,
    /// `(patch_rows, patch_cols)`.
    pub grid: (usize, usize),
    /// `num_patches / pooling_kernel_size²` — how many `image_token_id`
    /// placeholders the prompt must carry for this image.
    pub num_soft_tokens: usize,
}

/// Decode an encoded image and lay it out the way the vision tower expects.
///
/// Mirrors `Gemma4ImageProcessor._preprocess`: aspect-ratio-preserving bicubic
/// resize onto the patch/pooling grid, rescale to `[0, 1]` (Gemma 4 applies no
/// dataset mean/std — the `2·(x−0.5)` shift happens inside the tower), then
/// patchify in raster order with channel-last packing per pixel.
///
/// Padding to a fixed `max_patches` is intentionally omitted: it exists only to
/// batch differently-sized images, and a single image yields identical soft
/// tokens without it (asserted by `scripts/gemma4_vision_ref.py`).
///
/// Resampling note: upstream offers a torchvision path (`antialias=True`
/// bicubic) and a PIL path (`Image.BICUBIC`); this uses Catmull-Rom, which is
/// the same Keys cubic (a = −0.5) PIL uses. Sub-LSB pixel differences versus
/// torchvision are expected and are an input perturbation, not a port defect.
pub fn prepare_image(
    encoded: &[u8],
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<PreparedImage, String> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(format!(
            "max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {max_soft_tokens}"
        ));
    }
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;

    let img = decode_bounded(encoded)?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    let (target_h, target_w) =
        aspect_ratio_preserving_size(h, w, patch_size, max_patches, pooling_kernel_size)?;

    let resized = if target_h == h && target_w == w {
        rgb
    } else {
        image::imageops::resize(
            &rgb,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::CatmullRom,
        )
    };

    let rows = target_h / patch_size;
    let cols = target_w / patch_size;
    let per_patch = 3 * patch_size * patch_size;
    let mut pixel_values = vec![0.0f32; rows * cols * per_patch];

    // Layout mirrors upstream's `permute(1, 3, 2, 4, 0)`:
    // patch index is raster over (row, col); within a patch, raster over
    // (py, px) with the 3 channels innermost.
    let raw = resized.as_raw();
    for pr in 0..rows {
        for pc in 0..cols {
            let base = (pr * cols + pc) * per_patch;
            for py in 0..patch_size {
                let src_row = (pr * patch_size + py) * target_w;
                for px in 0..patch_size {
                    let src = (src_row + pc * patch_size + px) * 3;
                    let dst = base + (py * patch_size + px) * 3;
                    pixel_values[dst] = raw[src] as f32 / 255.0;
                    pixel_values[dst + 1] = raw[src + 1] as f32 / 255.0;
                    pixel_values[dst + 2] = raw[src + 2] as f32 / 255.0;
                }
            }
        }
    }

    Ok(PreparedImage {
        pixel_values,
        grid: (rows, cols),
        num_soft_tokens: (rows / pooling_kernel_size) * (cols / pooling_kernel_size),
    })
}

/// Largest aspect-ratio-preserving size that fits the patch budget and is
/// divisible by `pooling_kernel_size · patch_size` on both sides.
///
/// Mirrors `get_aspect_ratio_preserving_size` upstream.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("image has a zero dimension".to_string());
    }
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_h = ((factor * height as f64) / side_mult as f64).floor() as usize * side_mult;
    let mut target_w = ((factor * width as f64) / side_mult as f64).floor() as usize * side_mult;

    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 && target_w == 0 {
        return Err("image resized to 0×0".to_string());
    }
    if target_h == 0 {
        target_h = side_mult;
        target_w = ((width / height) * side_mult).min(max_side).max(side_mult);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = ((height / width) * side_mult).min(max_side).max(side_mult);
    }
    Ok((target_h, target_w))
}

#[cfg(test)]
mod tests {
    use super::aspect_ratio_preserving_size;

    #[test]
    fn landscape_fits_patch_budget_and_pooling_grid() {
        // 640×480 with the shipped defaults (patch 16, 280 soft tokens, pool 3)
        // is what scripts/gemma4_vision_ref.py reports as 912×672 → 57×42.
        let (h, w) = aspect_ratio_preserving_size(480, 640, 16, 280 * 9, 3).unwrap();
        assert_eq!((h, w), (672, 912));
        assert_eq!((h / 16) % 3, 0);
        assert_eq!((w / 16) % 3, 0);
        assert!((h / 16) * (w / 16) <= 280 * 9);
    }

    #[test]
    fn square_image_stays_square() {
        let (h, w) = aspect_ratio_preserving_size(1000, 1000, 16, 280 * 9, 3).unwrap();
        assert_eq!(h, w);
        assert_eq!(h % 48, 0);
        assert!((h / 16) * (w / 16) <= 280 * 9);
    }

    #[test]
    fn extreme_aspect_ratio_does_not_collapse() {
        let (h, w) = aspect_ratio_preserving_size(20, 4000, 16, 280 * 9, 3).unwrap();
        assert!(h >= 48 && w >= 48, "got {h}×{w}");
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(aspect_ratio_preserving_size(0, 100, 16, 2520, 3).is_err());
    }

    /// The decode limits are read from the environment on every call, and
    /// cargo runs tests in parallel, so anything that mutates them has to
    /// exclude the tests that depend on the defaults.
    static LIMIT_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::new(w, h);
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode png");
        out
    }

    /// The count has to be derivable from the header alone, because the server
    /// sizes prompts (context guard, usage) before deciding to decode anything.
    #[test]
    fn soft_token_count_matches_a_full_prepare() {
        let _guard = LIMIT_ENV.lock().unwrap_or_else(|e| e.into_inner());
        for (w, h) in [(640u32, 480u32), (300, 300), (1024, 100)] {
            let bytes = png(w, h);
            let counted = super::soft_token_count(&bytes, 16, 280, 3).expect("count");
            let prepared = super::prepare_image(&bytes, 16, 280, 3).expect("prepare");
            assert_eq!(
                counted, prepared.num_soft_tokens,
                "header-only count disagrees with the prepared grid for {w}×{h}"
            );
        }
    }

    /// A decompression bomb is a small file that expands into gigabytes of
    /// pixels. The header carries the dimensions, so it is rejected before
    /// anything is allocated for them.
    #[test]
    fn oversized_images_are_rejected_before_decode() {
        let _guard = LIMIT_ENV.lock().unwrap_or_else(|e| e.into_inner());
        // Bounds are read per call, so setting them here is enough.
        unsafe { std::env::set_var("LUMEN_VISION_MAX_IMAGE_PIXELS", "1024") };
        let bytes = png(256, 256); // 65536 pixels, well over the 1024 cap
        // `.err()` rather than `unwrap_err()`: the Ok variant carries a
        // multi-megabyte pixel buffer we do not want formatted into a panic.
        let err = super::prepare_image(&bytes, 16, 280, 3)
            .err()
            .expect("oversized image should be rejected");
        assert!(err.contains("pixel limit"), "unexpected error: {err}");
        assert!(super::soft_token_count(&bytes, 16, 280, 3).is_err());
        unsafe { std::env::remove_var("LUMEN_VISION_MAX_IMAGE_PIXELS") };
    }

    #[test]
    fn oversized_payloads_are_rejected_before_parsing() {
        let _guard = LIMIT_ENV.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("LUMEN_VISION_MAX_IMAGE_BYTES", "16") };
        let err = super::prepare_image(&png(32, 32), 16, 280, 3)
            .err()
            .expect("oversized payload should be rejected");
        assert!(err.contains("byte limit"), "unexpected error: {err}");
        unsafe { std::env::remove_var("LUMEN_VISION_MAX_IMAGE_BYTES") };
    }

    #[test]
    fn garbage_bytes_are_an_error_not_a_panic() {
        assert!(super::prepare_image(b"not an image at all", 16, 280, 3).is_err());
    }
}
