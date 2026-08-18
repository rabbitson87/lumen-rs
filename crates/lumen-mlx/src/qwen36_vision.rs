//! Native MLX Qwen 3.6 vision tower.
//!
//! Qwen 3.6's image encoder is the Qwen3-VL ViT — structurally unlike Gemma 4's
//! (see `gemma4_vision.rs`), so almost nothing is shared beyond the shape of
//! the problem:
//!   * `patch_embed`: one Conv3d whose kernel equals its stride, i.e. a plain
//!     linear over each `temporal × patch × patch × channel` block,
//!   * a **learned** `[num_position_embeddings, hidden]` position grid,
//!     bilinearly interpolated onto whatever patch grid this image produced,
//!   * `depth` blocks of **LayerNorm** (with bias) + fused QKV + 2-D RoPE +
//!     bidirectional attention + GELU MLP — not RMSNorm, not GeGLU,
//!   * a `spatial_merge_size²` patch merger that folds each 2×2 block of
//!     patches into one language-model token.
//!
//! Three conventions here are easy to get wrong and produce plausible-looking
//! activations rather than an error, so each is called out at its use site:
//! the token order is **merge-block major** (not raster), the block MLP uses
//! the tanh GELU approximation while the merger uses the exact erf GELU, and
//! the rotary table interleaves `(h, w)` frequencies before being duplicated
//! for `rotate_half`.
//!
//! Spec: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`
//! (`Qwen3VLVisionModel`), transformers 5.9.

#[cfg(feature = "mlx-native")]
pub mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use mlx_rs::Dtype;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};

    use std::sync::OnceLock;

    // ───────────────────────────── config ─────────────────────────────

    // The `vision_config` block is pure serde + arithmetic validation, so it
    // lives in the ungated `qwen35_config` module alongside the rest of this
    // config.json. Re-exported here so call sites are untouched.
    pub use crate::qwen35_config::NativeQwen36VisionConfig;

    /// LayerNorm epsilon. Hard-coded upstream (`nn.LayerNorm(..., eps=1e-6)`)
    /// rather than read from the config, so it is hard-coded here too.
    const LAYER_NORM_EPS: f32 = 1e-6;

    /// RoPE base for the vision tower. Also a literal upstream
    /// (`Qwen3VLVisionRotaryEmbedding(head_dim // 2)` takes `theta=10000.0` by
    /// default) — the text tower's much larger `rope_theta` does not apply.
    const VISION_ROPE_THETA: f32 = 10_000.0;

    /// Run the tower in float32 instead of the checkpoint's bf16.
    fn vision_f32_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("LUMEN_VISION_F32")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    /// Drain the lazy graph every N encoder blocks. Same rationale as the
    /// Gemma 4 tower: without it every block's activations stay live until the
    /// first eval and peak memory climbs by GBs on a long patch grid.
    fn vision_eval_every() -> usize {
        static EVERY: OnceLock<usize> = OnceLock::new();
        *EVERY.get_or_init(|| {
            std::env::var("LUMEN_VISION_EVAL_EVERY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
        })
    }

    // ───────────────────────────── weights ─────────────────────────────

    /// One encoder block. Projections are stored **pre-transposed** (`[in, out]`)
    /// so the forward path is a plain `matmul`; the tower runs `depth` times per
    /// image and transposes are not free.
    struct VisionBlock {
        norm1_w: Array,
        norm1_b: Array,
        norm2_w: Array,
        norm2_b: Array,
        qkv_t: Array,
        qkv_b: Array,
        proj_t: Array,
        proj_b: Array,
        fc1_t: Array,
        fc1_b: Array,
        fc2_t: Array,
        fc2_b: Array,
    }

    /// `merger`: LayerNorm over `hidden`, fold `merge²` patches together, then
    /// two linears down to the text model's `out_hidden_size`.
    struct VisionMerger {
        norm_w: Array,
        norm_b: Array,
        fc1_t: Array,
        fc1_b: Array,
        fc2_t: Array,
        fc2_b: Array,
    }

    pub struct NativeQwen36VisionTower {
        cfg: NativeQwen36VisionConfig,
        /// `[temporal·patch²·channels, hidden]`, pre-transposed and flattened
        /// from the `[hidden, T, ph, pw, C]` Conv3d kernel.
        patch_proj_t: Array,
        patch_proj_b: Array,
        /// `[num_position_embeddings, hidden]` learned grid.
        pos_embed: Array,
        blocks: Vec<VisionBlock>,
        merger: VisionMerger,
        dtype: Dtype,
    }

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

    /// Look up an `[out, in]` linear weight and return it as `[in, out]`.
    fn take_t(
        weights: &std::collections::HashMap<String, Array>,
        name: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        let w = take(weights, name, dtype)?;
        mlx_rs::ops::transpose(&w).with_context(|| format!("transpose(`{name}`)"))
    }

    impl NativeQwen36VisionTower {
        /// Build the tower from the raw (un-sanitized) weight bag.
        ///
        /// Call this **before** `NativeWeights::sanitize()`, which drops every
        /// `vision_tower.*` / `model.visual.*` entry.
        pub fn load(
            weights: &std::collections::HashMap<String, Array>,
            cfg: NativeQwen36VisionConfig,
        ) -> Result<Self> {
            cfg.validate()?;
            let dtype = if vision_f32_enabled() {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            };

            // `patch_embed.proj` is a Conv3d whose kernel equals its stride and
            // whose input is exactly one kernel, so it is a linear map. MLX
            // stores the kernel as `[out, kT, kH, kW, in]` (channel last),
            // which fixes the feature order of a patch row to
            // `(t, py, px, c)` — see `patchify` below, which emits that order
            // directly rather than permuting at runtime.
            let proj = take(weights, "vision_tower.patch_embed.proj.weight", dtype)?;
            let expect = [
                cfg.hidden_size as i32,
                cfg.temporal_patch_size as i32,
                cfg.patch_size as i32,
                cfg.patch_size as i32,
                cfg.in_channels as i32,
            ];
            if proj.shape() != expect {
                return Err(anyhow!(
                    "patch_embed.proj.weight is {:?}, expected {expect:?} \
                     ([out, kT, kH, kW, in] — a torch-layout checkpoint would be [out, in, kT, kH, kW])",
                    proj.shape()
                ));
            }
            let patch_in =
                (cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size * cfg.in_channels)
                    as i32;
            let proj = mlx_rs::ops::reshape(&proj, &[cfg.hidden_size as i32, patch_in])
                .context("flatten patch_embed kernel")?;
            let patch_proj_t = mlx_rs::ops::transpose(&proj).context("transpose patch_embed")?;

            let pos_embed = take(weights, "vision_tower.pos_embed.weight", dtype)?;
            if pos_embed.shape() != [cfg.num_position_embeddings as i32, cfg.hidden_size as i32] {
                return Err(anyhow!(
                    "pos_embed.weight is {:?}, expected [{}, {}]",
                    pos_embed.shape(),
                    cfg.num_position_embeddings,
                    cfg.hidden_size
                ));
            }

            let mut blocks = Vec::with_capacity(cfg.depth);
            for i in 0..cfg.depth {
                let p = format!("vision_tower.blocks.{i}.");
                blocks.push(VisionBlock {
                    norm1_w: take(weights, &format!("{p}norm1.weight"), dtype)?,
                    norm1_b: take(weights, &format!("{p}norm1.bias"), dtype)?,
                    norm2_w: take(weights, &format!("{p}norm2.weight"), dtype)?,
                    norm2_b: take(weights, &format!("{p}norm2.bias"), dtype)?,
                    qkv_t: take_t(weights, &format!("{p}attn.qkv.weight"), dtype)?,
                    qkv_b: take(weights, &format!("{p}attn.qkv.bias"), dtype)?,
                    proj_t: take_t(weights, &format!("{p}attn.proj.weight"), dtype)?,
                    proj_b: take(weights, &format!("{p}attn.proj.bias"), dtype)?,
                    fc1_t: take_t(weights, &format!("{p}mlp.linear_fc1.weight"), dtype)?,
                    fc1_b: take(weights, &format!("{p}mlp.linear_fc1.bias"), dtype)?,
                    fc2_t: take_t(weights, &format!("{p}mlp.linear_fc2.weight"), dtype)?,
                    fc2_b: take(weights, &format!("{p}mlp.linear_fc2.bias"), dtype)?,
                });
            }

            let merger = VisionMerger {
                norm_w: take(weights, "vision_tower.merger.norm.weight", dtype)?,
                norm_b: take(weights, "vision_tower.merger.norm.bias", dtype)?,
                fc1_t: take_t(weights, "vision_tower.merger.linear_fc1.weight", dtype)?,
                fc1_b: take(weights, "vision_tower.merger.linear_fc1.bias", dtype)?,
                fc2_t: take_t(weights, "vision_tower.merger.linear_fc2.weight", dtype)?,
                fc2_b: take(weights, "vision_tower.merger.linear_fc2.bias", dtype)?,
            };

            Ok(Self {
                cfg,
                patch_proj_t,
                patch_proj_b: take(weights, "vision_tower.patch_embed.proj.bias", dtype)?,
                pos_embed,
                blocks,
                merger,
                dtype,
            })
        }

        pub fn config(&self) -> &NativeQwen36VisionConfig {
            &self.cfg
        }

        /// Encode one image into `[num_tokens, out_hidden_size]` embeddings.
        ///
        /// * `patches` — `[grid_h · grid_w, temporal·patch²·channels]` in
        ///   **merge-block order** (see [`super::patchify`]).
        /// * `grid` — `(grid_h, grid_w)` in patches; both must be multiples of
        ///   `spatial_merge_size`.
        ///
        /// Returns `grid_h · grid_w / merge²` rows — one per language-model
        /// image token.
        pub fn forward(&self, patches: &Array, grid: (usize, usize)) -> Result<Array> {
            let (gh, gw) = grid;
            let merge = self.cfg.spatial_merge_size;
            let n = (gh * gw) as i32;
            let hidden = self.cfg.hidden_size as i32;

            if patches.ndim() != 2 {
                return Err(anyhow!(
                    "patches must be [N, T·P²·C], got {:?}",
                    patches.shape()
                ));
            }
            if patches.shape()[0] != n {
                return Err(anyhow!(
                    "patches has {} rows but grid {gh}×{gw} implies {n}",
                    patches.shape()[0]
                ));
            }
            if gh % merge != 0 || gw % merge != 0 {
                return Err(anyhow!(
                    "patch grid {gh}×{gw} is not divisible by spatial_merge_size {merge}"
                ));
            }

            // ── patch embed ──
            let px = patches.as_dtype(self.dtype).context("cast patches")?;
            let mut h =
                mlx_rs::ops::matmul(&px, &self.patch_proj_t).context("patch_embed: matmul")?;
            h = mlx_rs::ops::add(&h, &self.patch_proj_b).context("patch_embed: + bias")?;

            // ── learned position grid, interpolated onto this image ──
            h = mlx_rs::ops::add(&h, &self.interpolated_pos_embed(gh, gw)?)
                .context("patch_embed: + interpolated position embedding")?;

            // ── 2-D RoPE tables, shared by every block ──
            let (cos, sin) = self.build_rope(gh, gw)?;

            // ── encoder ──
            let mut h =
                mlx_rs::ops::reshape(&h, &[1, n, hidden]).context("reshape to [1, N, hidden]")?;
            let eval_every = vision_eval_every();
            for (idx, blk) in self.blocks.iter().enumerate() {
                h = self
                    .block_forward(&h, blk, &cos, &sin, n)
                    .with_context(|| format!("vision block {idx}"))?;
                if eval_every > 0 && (idx + 1) % eval_every == 0 {
                    h.eval().context("vision encoder: periodic eval")?;
                }
            }

            // ── merger ──
            // LayerNorm runs on the un-folded `hidden`-wide tokens
            // (`use_postshuffle_norm=False`); only then do `merge²` consecutive
            // patches — which the merge-block token order made adjacent —
            // concatenate into one row.
            let h = mlx_rs::ops::reshape(&h, &[n, hidden]).context("merger: flatten")?;
            let h = layer_norm(&h, &self.merger.norm_w, &self.merger.norm_b)
                .context("merger: layer_norm")?;
            let merged_dim = hidden * self.cfg.merge_unit() as i32;
            let h = mlx_rs::ops::reshape(&h, &[n / self.cfg.merge_unit() as i32, merged_dim])
                .context("merger: fold merge blocks")?;
            let h = mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&h, &self.merger.fc1_t).context("merger: fc1")?,
                &self.merger.fc1_b,
            )
            .context("merger: fc1 + bias")?;
            // The merger uses `nn.GELU()` — the exact erf form — while the
            // block MLPs use the tanh approximation. Mixing them up shifts
            // every soft token by a small, entirely silent amount.
            let h = gelu_erf(&h).context("merger: gelu")?;
            mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&h, &self.merger.fc2_t).context("merger: fc2")?,
                &self.merger.fc2_b,
            )
            .context("merger: fc2 + bias")
        }

        /// Bilinearly resample the learned `side × side` position grid onto a
        /// `gh × gw` patch grid, returning `[gh·gw, hidden]` in merge-block
        /// order.
        ///
        /// Mirrors `fast_pos_embed_interpolate`: sample points are
        /// `linspace(0, side - 1, n)`, the four neighbours are gathered with
        /// their bilinear weights, and the result is permuted into merge-block
        /// order to match the token layout.
        fn interpolated_pos_embed(&self, gh: usize, gw: usize) -> Result<Array> {
            let hidden = self.cfg.hidden_size as i32;
            let n = gh * gw;
            let (idx, wgt) = interpolation_plan(
                gh,
                gw,
                self.cfg.grid_per_side(),
                self.cfg.spatial_merge_size,
            );

            let mut acc: Option<Array> = None;
            for k in 0..4 {
                let ids: Vec<i32> = idx.iter().map(|c| c[k]).collect();
                let ws: Vec<f32> = wgt.iter().map(|c| c[k]).collect();
                let ids = Array::from_slice(&ids, &[n as i32]);
                let rows = self
                    .pos_embed
                    .take_axis(&ids, 0)
                    .context("pos_embed: gather")?;
                let w = Array::from_slice(&ws, &[n as i32, 1])
                    .as_dtype(self.dtype)
                    .context("pos_embed: cast weights")?;
                let term = mlx_rs::ops::multiply(&rows, &w).context("pos_embed: weight")?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => mlx_rs::ops::add(&a, &term).context("pos_embed: accumulate")?,
                });
            }
            let out = acc.ok_or_else(|| anyhow!("pos_embed: no corners"))?;
            debug_assert_eq!(out.shape(), [n as i32, hidden]);
            Ok(out)
        }

        /// `(cos, sin)` of shape `[N, head_dim]` for the 2-D rotary.
        ///
        /// Upstream builds a `[max_hw, head_dim/4]` frequency table, looks it
        /// up at each token's `(row, col)`, flattens the pair to `head_dim/2`,
        /// then concatenates that with itself so `rotate_half` pairs channel
        /// `c` with `c + head_dim/2` — i.e. the same frequency of the same
        /// axis. Positions are the token's full-resolution `(row, col)`, and
        /// tokens arrive in merge-block order.
        fn build_rope(&self, gh: usize, gw: usize) -> Result<(Array, Array)> {
            let head_dim = self.cfg.head_dim();
            let n = gh * gw;
            let angles = rope_angles(gh, gw, head_dim, self.cfg.spatial_merge_size);
            let angles = Array::from_slice(&angles, &[n as i32, head_dim as i32]);
            let cos = mlx_rs::ops::cos(&angles)
                .context("rope: cos")?
                .as_dtype(self.dtype)?;
            let sin = mlx_rs::ops::sin(&angles)
                .context("rope: sin")?
                .as_dtype(self.dtype)?;
            Ok((cos, sin))
        }

        fn block_forward(
            &self,
            h: &Array,
            blk: &VisionBlock,
            cos: &Array,
            sin: &Array,
            n: i32,
        ) -> Result<Array> {
            let hidden = self.cfg.hidden_size as i32;
            let heads = self.cfg.num_heads as i32;
            let head_dim = self.cfg.head_dim() as i32;

            // ── attention ──
            let residual = h.clone();
            let x = mlx_rs::ops::reshape(h, &[n, hidden]).context("flatten for norm1")?;
            let x = layer_norm(&x, &blk.norm1_w, &blk.norm1_b).context("norm1")?;

            let qkv = mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&x, &blk.qkv_t).context("qkv")?,
                &blk.qkv_b,
            )
            .context("qkv + bias")?;
            // `[N, 3·hidden]` → `[N, 3, heads, head_dim]`, then split on axis 1.
            let qkv =
                mlx_rs::ops::reshape(&qkv, &[n, 3, heads, head_dim]).context("reshape qkv")?;
            let q = qkv.index((Ellipsis, 0, .., ..));
            let k = qkv.index((Ellipsis, 1, .., ..));
            let v = qkv.index((Ellipsis, 2, .., ..));

            // cos/sin are `[N, head_dim]`; broadcast over heads.
            let c = mlx_rs::ops::reshape(cos, &[n, 1, head_dim]).context("reshape cos")?;
            let s = mlx_rs::ops::reshape(sin, &[n, 1, head_dim]).context("reshape sin")?;
            let q = apply_rotary(&q, &c, &s, head_dim).context("rotary q")?;
            let k = apply_rotary(&k, &c, &s, head_dim).context("rotary k")?;

            // `[N, heads, head_dim]` → `[1, heads, N, head_dim]` for SDPA.
            let to_bhnd = |t: &Array, name: &str| -> Result<Array> {
                let t = mlx_rs::ops::reshape(t, &[1, n, heads, head_dim])
                    .with_context(|| format!("{name}: add batch axis"))?;
                mlx_rs::ops::transpose_axes(&t, &[0, 2, 1, 3])
                    .with_context(|| format!("{name}: transpose to [B, H, N, D]"))
            };
            let (q, k, v) = (to_bhnd(&q, "q")?, to_bhnd(&k, "k")?, to_bhnd(&v, "v")?);

            // Bidirectional: a single image is one packed segment, so there is
            // nothing to mask (upstream's `cu_seqlens` has one entry per image
            // and we encode one at a time).
            let scale = (head_dim as f32).powf(-0.5);
            let attn = crate::native_attention::sdpa(&q, &k, &v, scale, false).context("sdpa")?;
            let attn =
                mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3]).context("transpose attn back")?;
            let attn = mlx_rs::ops::reshape(&attn, &[n, hidden]).context("flatten attn")?;
            let attn = mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&attn, &blk.proj_t).context("attn proj")?,
                &blk.proj_b,
            )
            .context("attn proj + bias")?;
            let attn = mlx_rs::ops::reshape(&attn, &[1, n, hidden]).context("reshape attn out")?;
            let h = mlx_rs::ops::add(&residual, &attn).context("+residual (attn)")?;

            // ── MLP ──
            let residual = h.clone();
            let x = mlx_rs::ops::reshape(&h, &[n, hidden]).context("flatten for norm2")?;
            let x = layer_norm(&x, &blk.norm2_w, &blk.norm2_b).context("norm2")?;
            let x = mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&x, &blk.fc1_t).context("fc1")?,
                &blk.fc1_b,
            )
            .context("fc1 + bias")?;
            // Block MLPs use `hidden_act = gelu_pytorch_tanh`; the merger does
            // not. See the merger for the other half of this trap.
            let x = gelu_tanh(&x).context("mlp gelu")?;
            let x = mlx_rs::ops::add(
                &mlx_rs::ops::matmul(&x, &blk.fc2_t).context("fc2")?,
                &blk.fc2_b,
            )
            .context("fc2 + bias")?;
            let x = mlx_rs::ops::reshape(&x, &[1, n, hidden]).context("reshape mlp out")?;
            mlx_rs::ops::add(&residual, &x).context("+residual (mlp)")
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    /// Rotary angles for every token, `[N · head_dim]` row-major.
    ///
    /// Layout per token: `[row·f₀ … row·f_{k-1} | col·f₀ … col·f_{k-1}]`
    /// repeated twice, where `k = head_dim / 4`. The repeat is what makes
    /// `rotate_half` — which pairs channel `c` with `c + head_dim/2` — pair a
    /// frequency with *itself on the same axis*; a single copy would pair row
    /// frequencies against column frequencies and quietly destroy the 2-D
    /// structure. Tokens are emitted in merge-block order to match
    /// [`super::patchify`].
    ///
    /// Kept free of MLX so the layout, which has no runtime error signal, is
    /// unit-testable on any machine.
    pub(crate) fn rope_angles(gh: usize, gw: usize, head_dim: usize, merge: usize) -> Vec<f32> {
        // `Qwen3VLVisionRotaryEmbedding(head_dim // 2)` → `arange(0, dim, 2)`.
        let rot_dim = head_dim / 2;
        let inv: Vec<f32> = (0..rot_dim / 2)
            .map(|i| 1.0 / VISION_ROPE_THETA.powf((2 * i) as f32 / rot_dim as f32))
            .collect();

        let mut angles: Vec<f32> = Vec::with_capacity(gh * gw * head_dim);
        for br in 0..gh / merge {
            for bc in 0..gw / merge {
                for ir in 0..merge {
                    for ic in 0..merge {
                        let (r, c) = ((br * merge + ir) as f32, (bc * merge + ic) as f32);
                        for _ in 0..2 {
                            for f in &inv {
                                angles.push(r * f);
                            }
                            for f in &inv {
                                angles.push(c * f);
                            }
                        }
                    }
                }
            }
        }
        angles
    }

    /// Bilinear sampling plan for the learned position grid: for each token in
    /// merge-block order, the four neighbouring grid indices and their weights.
    ///
    /// Pure arithmetic, separated from the gather so the sampling — which is
    /// where an off-by-one silently blurs or shifts the whole grid — can be
    /// checked without weights or a Metal device.
    pub(crate) fn interpolation_plan(
        gh: usize,
        gw: usize,
        side: usize,
        merge: usize,
    ) -> (Vec<[i32; 4]>, Vec<[f32; 4]>) {
        let axis = |n: usize| -> (Vec<i32>, Vec<i32>, Vec<f32>) {
            let (mut lo, mut hi, mut frac) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..n {
                // `linspace(0, side - 1, n)`; a single sample sits at 0.
                let x = if n > 1 {
                    (side - 1) as f32 * i as f32 / (n - 1) as f32
                } else {
                    0.0
                };
                // `int()` truncates toward zero, and x >= 0 here.
                let f = x as i32;
                lo.push(f);
                hi.push((f + 1).min(side as i32 - 1));
                frac.push(x - f as f32);
            }
            (lo, hi, frac)
        };
        let (h_lo, h_hi, dh) = axis(gh);
        let (w_lo, w_hi, dw) = axis(gw);

        let n = gh * gw;
        let (mut idx, mut wgt) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for br in 0..gh / merge {
            for bc in 0..gw / merge {
                for ir in 0..merge {
                    for ic in 0..merge {
                        let (r, c) = (br * merge + ir, bc * merge + ic);
                        let (rl, rh, fr) = (h_lo[r], h_hi[r], dh[r]);
                        let (cl, ch, fc) = (w_lo[c], w_hi[c], dw[c]);
                        let s = side as i32;
                        idx.push([rl * s + cl, rl * s + ch, rh * s + cl, rh * s + ch]);
                        wgt.push([
                            (1.0 - fr) * (1.0 - fc),
                            (1.0 - fr) * fc,
                            fr * (1.0 - fc),
                            fr * fc,
                        ]);
                    }
                }
            }
        }
        (idx, wgt)
    }

    /// LayerNorm over the last axis with the tower's fixed epsilon.
    fn layer_norm(x: &Array, weight: &Array, bias: &Array) -> Result<Array> {
        mlx_rs::fast::layer_norm(x, Some(weight), Some(bias), LAYER_NORM_EPS)
            .context("mlx-rs fast::layer_norm FFI call failed")
    }

    /// `x·cos + rotate_half(x)·sin` over the last axis.
    fn apply_rotary(x: &Array, cos: &Array, sin: &Array, d: i32) -> Result<Array> {
        let half = d / 2;
        let x1 = x.index((Ellipsis, 0..half));
        let x2 = x.index((Ellipsis, half..d));
        let rotated = mlx_rs::ops::concatenate_axis(
            &[
                &mlx_rs::ops::negative(&x2).context("rotate_half: negate")?,
                &x1,
            ],
            -1,
        )
        .context("rotate_half: concat")?;
        mlx_rs::ops::add(
            &mlx_rs::ops::multiply(x, cos).context("rotary: x·cos")?,
            &mlx_rs::ops::multiply(&rotated, sin).context("rotary: rot·sin")?,
        )
        .context("rotary: x·cos + rot·sin")
    }

    /// GELU, tanh approximation (`gelu_pytorch_tanh`). Used by the block MLPs.
    fn gelu_tanh(x: &Array) -> Result<Array> {
        let dt = x.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.797_884_6_f32).as_dtype(dt)?;
        let x_sq = mlx_rs::ops::multiply(x, x)?;
        let x_cubed = mlx_rs::ops::multiply(&x_sq, x)?;
        let inner = mlx_rs::ops::add(x, &mlx_rs::ops::multiply(&x_cubed, &c3)?)?;
        let t = mlx_rs::ops::tanh(&mlx_rs::ops::multiply(&coeff, &inner)?)?;
        mlx_rs::ops::multiply(
            &mlx_rs::ops::multiply(&half, x)?,
            &mlx_rs::ops::add(&one, &t)?,
        )
        .context("gelu_tanh")
    }

    /// GELU, exact erf form (`nn.GELU()`). Used by the merger only.
    fn gelu_erf(x: &Array) -> Result<Array> {
        let dt = x.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let inv_sqrt2 = Array::from_f32(std::f32::consts::FRAC_1_SQRT_2).as_dtype(dt)?;
        let e = mlx_rs::ops::erf(&mlx_rs::ops::multiply(x, &inv_sqrt2)?).context("erf")?;
        mlx_rs::ops::multiply(
            &mlx_rs::ops::multiply(&half, x)?,
            &mlx_rs::ops::add(&one, &e)?,
        )
        .context("gelu_erf")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{NativeQwen36VisionConfig, NativeQwen36VisionTower};

// ─────────────────────────── image preprocessing ───────────────────────────
//
// Pure CPU, no MLX — mirrors `Qwen2VLImageProcessor` as Qwen3-VL reuses it.

/// Result of preparing one image for the vision tower.
pub struct PreparedImage {
    /// `[grid_h · grid_w × (temporal·patch²·channels)]` in **merge-block**
    /// order, rescaled and normalized.
    pub patches: Vec<f32>,
    /// `(grid_h, grid_w)` in patches.
    pub grid: (usize, usize),
    /// `grid_h · grid_w / merge²` — how many `image_pad` placeholders the
    /// prompt must carry for this image.
    pub num_tokens: usize,
}

use crate::vision_image::{decode_bounded, env_usize};

/// CLIP-style normalization constants Qwen's processor uses.
const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Round `value` to the nearest multiple of `factor`.
fn round_by_factor(value: f64, factor: usize) -> usize {
    ((value / factor as f64).round() as usize).max(1) * factor
}
fn floor_by_factor(value: f64, factor: usize) -> usize {
    (value / factor as f64).floor() as usize * factor
}
fn ceil_by_factor(value: f64, factor: usize) -> usize {
    (value / factor as f64).ceil() as usize * factor
}

/// Qwen's `smart_resize`: keep the aspect ratio within `MAX_RATIO`, land both
/// sides on a multiple of `factor` (= `patch_size · merge_size`), and keep the
/// total pixel count inside `[min_pixels, max_pixels]`.
///
/// Returns `(height, width)`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    const MAX_RATIO: f64 = 200.0;
    if height == 0 || width == 0 {
        return Err("image has a zero dimension".to_string());
    }
    let (hf, wf) = (height as f64, width as f64);
    if hf.max(wf) / hf.min(wf) > MAX_RATIO {
        return Err(format!(
            "aspect ratio {:.1} exceeds the {MAX_RATIO} limit ({height}×{width})",
            hf.max(wf) / hf.min(wf)
        ));
    }
    let mut h = round_by_factor(hf, factor).max(factor);
    let mut w = round_by_factor(wf, factor).max(factor);
    if h * w > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h = floor_by_factor(hf / beta, factor).max(factor);
        w = floor_by_factor(wf / beta, factor).max(factor);
    } else if h * w < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h = ceil_by_factor(hf * beta, factor);
        w = ceil_by_factor(wf * beta, factor);
    }
    Ok((h, w))
}

/// Lay out a resized RGB image as patch rows in merge-block order.
///
/// Two orderings matter and neither has an error signal if you get it wrong:
///
/// * **Rows** are grouped by `merge_size × merge_size` block — block row, block
///   column, then the patches inside that block — *not* raster. The tower's
///   merger folds `merge²` consecutive rows into one token, and the rotary
///   table is built in the same order, so raster order silently scrambles both.
/// * **Within a row** the features run `(t, py, px, c)` — channel innermost.
///   That matches the MLX `[out, kT, kH, kW, in]` Conv3d kernel directly; the
///   torch checkpoint's `[out, in, kT, kH, kW]` would want `(c, t, py, px)`.
///
/// A still image is duplicated across the `temporal_patch_size` frames, which
/// is what the processor does when it pads a single frame to the temporal
/// kernel depth.
#[allow(clippy::too_many_arguments)]
pub fn patchify(
    rgb: &[u8],
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
) -> Vec<f32> {
    let (gh, gw) = (height / patch_size, width / patch_size);
    let per_patch = temporal_patch_size * patch_size * patch_size * 3;
    let mut out = vec![0.0f32; gh * gw * per_patch];

    let mut row = 0usize;
    for br in 0..gh / merge_size {
        for bc in 0..gw / merge_size {
            for ir in 0..merge_size {
                for ic in 0..merge_size {
                    let (pr, pc) = (br * merge_size + ir, bc * merge_size + ic);
                    let base = row * per_patch;
                    for t in 0..temporal_patch_size {
                        let t_off = base + t * patch_size * patch_size * 3;
                        for py in 0..patch_size {
                            let src_row = (pr * patch_size + py) * width;
                            for px in 0..patch_size {
                                let src = (src_row + pc * patch_size + px) * 3;
                                let dst = t_off + (py * patch_size + px) * 3;
                                for c in 0..3 {
                                    let v = rgb[src + c] as f32 / 255.0;
                                    out[dst + c] = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    out
}

/// The two pieces of tower arithmetic with no runtime error signal: a wrong
/// rotary layout or a shifted position-grid sampling both yield plausible
/// activations. Verified by their defining properties rather than against a
/// dumped reference tensor, so the checks run anywhere, with no Python and no
/// checkpoint.
#[cfg(all(test, feature = "mlx-native"))]
mod geometry_tests {
    use super::imp::{interpolation_plan, rope_angles};

    const HEAD_DIM: usize = 72; // 1152 / 16
    const MERGE: usize = 2;

    /// Row-major `[N, head_dim]` view of the angle table.
    fn token(angles: &[f32], i: usize) -> &[f32] {
        &angles[i * HEAD_DIM..(i + 1) * HEAD_DIM]
    }

    /// `rotate_half` pairs channel `c` with `c + head_dim/2`. For the rotation
    /// to be a rotation, both members of a pair must carry the *same* angle —
    /// that is exactly what duplicating the `[row | col]` table achieves, and
    /// dropping the duplicate is the classic way to break a 2-D vision rotary
    /// without breaking anything visible.
    #[test]
    fn rotate_half_pairs_carry_matching_angles() {
        let angles = rope_angles(4, 6, HEAD_DIM, MERGE);
        for i in 0..24 {
            let t = token(&angles, i);
            for c in 0..HEAD_DIM / 2 {
                assert_eq!(
                    t[c],
                    t[c + HEAD_DIM / 2],
                    "token {i}: channel {c} and its rotate_half partner disagree"
                );
            }
        }
    }

    /// The first quarter of the channels is driven by the row coordinate and
    /// the second by the column. Swapping them transposes every image.
    #[test]
    fn row_and_column_drive_their_own_channel_halves() {
        let angles = rope_angles(4, 4, HEAD_DIM, MERGE);
        let quarter = HEAD_DIM / 4;
        // Merge-block order: rows 0..4 are block (0,0) → patches (0,0) (0,1)
        // (1,0) (1,1).
        let p00 = token(&angles, 0);
        let p01 = token(&angles, 1); // same row, next column
        let p10 = token(&angles, 2); // next row, same column
        // Row half identical between (0,0) and (0,1); column half differs.
        assert_eq!(&p00[..quarter], &p01[..quarter]);
        assert_ne!(&p00[quarter..2 * quarter], &p01[quarter..2 * quarter]);
        // …and the mirror image for (1,0).
        assert_ne!(&p00[..quarter], &p10[..quarter]);
        assert_eq!(&p00[quarter..2 * quarter], &p10[quarter..2 * quarter]);
    }

    /// RoPE's defining property: after rotation, `q·k` depends only on the
    /// *relative* position. Checked here on the angle table directly — pairs of
    /// tokens with equal `(Δrow, Δcol)` must have equal angle differences.
    #[test]
    fn angles_depend_only_on_relative_position() {
        let (gh, gw) = (4usize, 4usize);
        let angles = rope_angles(gh, gw, HEAD_DIM, MERGE);
        // Token index → (row, col) under merge-block order.
        let coord = |i: usize| -> (usize, usize) {
            let per_block = MERGE * MERGE;
            let (b, within) = (i / per_block, i % per_block);
            let (br, bc) = (b / (gw / MERGE), b % (gw / MERGE));
            (br * MERGE + within / MERGE, bc * MERGE + within % MERGE)
        };
        let diff = |a: usize, b: usize| -> Vec<f32> {
            token(&angles, b)
                .iter()
                .zip(token(&angles, a))
                .map(|(x, y)| x - y)
                .collect()
        };

        // Every token pair offset by (+1, +1) must produce the same angle
        // delta, whichever part of the grid it sits in.
        let mut reference: Option<(usize, usize, Vec<f32>)> = None;
        let mut checked = 0;
        for a in 0..gh * gw {
            for b in 0..gh * gw {
                let (ra, ca) = coord(a);
                let (rb, cb) = coord(b);
                if rb != ra + 1 || cb != ca + 1 {
                    continue;
                }
                let d = diff(a, b);
                match &reference {
                    None => reference = Some((a, b, d)),
                    Some((ra0, rb0, want)) => {
                        for (i, (x, y)) in d.iter().zip(want.iter()).enumerate() {
                            assert!(
                                (x - y).abs() < 1e-6,
                                "channel {i}: pair ({a},{b}) delta {x} != pair ({ra0},{rb0}) delta {y} \
                                 — angles are not translation-invariant"
                            );
                        }
                    }
                }
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "expected several (+1,+1) token pairs on a {gh}×{gw} grid, saw {checked}"
        );
        // Sanity: the deltas are not all zero, i.e. position actually matters.
        let (_, _, want) = reference.expect("at least one pair");
        assert!(
            want.iter().any(|d| d.abs() > 1e-6),
            "angle deltas are all zero — positions are not being applied"
        );
    }

    /// Interpolation weights are a partition of unity — otherwise the position
    /// embedding is silently scaled up or down.
    #[test]
    fn interpolation_weights_sum_to_one() {
        for (gh, gw) in [(4usize, 6usize), (48, 48), (2, 96)] {
            let (_, wgt) = interpolation_plan(gh, gw, 48, MERGE);
            assert_eq!(wgt.len(), gh * gw);
            for (i, w) in wgt.iter().enumerate() {
                let s: f32 = w.iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-5,
                    "{gh}×{gw} token {i}: weights sum to {s}"
                );
                assert!(w.iter().all(|&x| (-1e-6..=1.0 + 1e-6).contains(&x)));
            }
        }
    }

    /// A grid that matches the learned side length exactly must sample it
    /// one-to-one: no blending, no shift.
    #[test]
    fn native_resolution_grid_samples_the_table_exactly() {
        const SIDE: usize = 48;
        let (idx, wgt) = interpolation_plan(SIDE, SIDE, SIDE, MERGE);
        for (i, (ix, w)) in idx.iter().zip(wgt.iter()).enumerate() {
            let dominant = w.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                (dominant - 1.0).abs() < 1e-5,
                "token {i} blended ({w:?}) on a native-resolution grid"
            );
            // …and the corner it lands on is the patch's own grid cell.
            let per_block = MERGE * MERGE;
            let (b, within) = (i / per_block, i % per_block);
            let (br, bc) = (b / (SIDE / MERGE), b % (SIDE / MERGE));
            let (r, c) = (br * MERGE + within / MERGE, bc * MERGE + within % MERGE);
            assert_eq!(
                ix[0],
                (r * SIDE + c) as i32,
                "token {i} sampled the wrong cell"
            );
        }
    }

    /// Sampling spans the whole table: the first token reads the top-left cell
    /// and the last reads the bottom-right, so the grid is stretched over the
    /// image rather than cropped from a corner.
    #[test]
    fn sampling_covers_the_full_learned_grid() {
        const SIDE: usize = 48;
        let (gh, gw) = (8usize, 12usize);
        let (idx, _) = interpolation_plan(gh, gw, SIDE, MERGE);
        assert_eq!(idx[0][0], 0, "first token should start at the grid origin");
        let max = idx.iter().flat_map(|c| c.iter()).copied().max().unwrap();
        assert_eq!(
            max,
            (SIDE * SIDE - 1) as i32,
            "sampling should reach the far corner of the learned grid"
        );
    }
}

/// Per-image token budget, in **merged** tokens. Overridable with
/// `LUMEN_VISION_MAX_IMAGE_TOKENS`; the default keeps a single image well under
/// a thousand prompt tokens, which is what a 36 GB box wants.
const DEFAULT_MAX_IMAGE_TOKENS: usize = 1024;
/// Floor, so a thumbnail still gets enough patches to be legible.
const DEFAULT_MIN_IMAGE_TOKENS: usize = 16;

/// Decode an encoded image and lay it out for the tower.
///
/// Resizing targets a **token** budget rather than a pixel one, because tokens
/// are what the prompt and the KV cache pay for: one merged token covers
/// `merge² · patch²` pixels, so the budget converts directly.
#[cfg(feature = "mlx-native")]
pub fn prepare_image(
    encoded: &[u8],
    cfg: &NativeQwen36VisionConfig,
) -> Result<PreparedImage, String> {
    let factor = cfg.patch_size * cfg.spatial_merge_size;
    let px_per_token = factor * factor;
    let max_tokens = env_usize("LUMEN_VISION_MAX_IMAGE_TOKENS", DEFAULT_MAX_IMAGE_TOKENS);
    let min_tokens =
        env_usize("LUMEN_VISION_MIN_IMAGE_TOKENS", DEFAULT_MIN_IMAGE_TOKENS).min(max_tokens);

    let img = decode_bounded(encoded)?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    let (target_h, target_w) = smart_resize(
        h,
        w,
        factor,
        min_tokens * px_per_token,
        max_tokens * px_per_token,
    )?;
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

    let grid = (target_h / cfg.patch_size, target_w / cfg.patch_size);
    let patches = patchify(
        resized.as_raw(),
        target_h,
        target_w,
        cfg.patch_size,
        cfg.temporal_patch_size,
        cfg.spatial_merge_size,
    );
    Ok(PreparedImage {
        patches,
        num_tokens: (grid.0 / cfg.spatial_merge_size) * (grid.1 / cfg.spatial_merge_size),
        grid,
    })
}

/// Merged `(h, w)` grid — what the tower emits and what the MRoPE positions
/// are laid out over.
impl PreparedImage {
    pub fn merged_grid(&self, merge_size: usize) -> (usize, usize) {
        (self.grid.0 / merge_size, self.grid.1 / merge_size)
    }
}

/// Prompt tokens this image will occupy: its merged-token run plus the
/// `<|vision_start|>` / `<|vision_end|>` sentinels.
///
/// Header-only — the token count follows from the image's dimensions alone, so
/// the server can size a prompt for the context guard and usage accounting
/// without paying for a decode it may then throw away.
#[cfg(feature = "mlx-native")]
pub fn image_prompt_tokens(
    encoded: &[u8],
    cfg: &NativeQwen36VisionConfig,
) -> Result<usize, String> {
    let factor = cfg.patch_size * cfg.spatial_merge_size;
    let px_per_token = factor * factor;
    let max_tokens = env_usize("LUMEN_VISION_MAX_IMAGE_TOKENS", DEFAULT_MAX_IMAGE_TOKENS);
    let min_tokens =
        env_usize("LUMEN_VISION_MIN_IMAGE_TOKENS", DEFAULT_MIN_IMAGE_TOKENS).min(max_tokens);
    let (w, h) = crate::vision_image::dimensions_bounded(encoded)?;
    let (th, tw) = smart_resize(
        h as usize,
        w as usize,
        factor,
        min_tokens * px_per_token,
        max_tokens * px_per_token,
    )?;
    let merged = (th / factor) * (tw / factor);
    // + `<|vision_start|>` and `<|vision_end|>`.
    Ok(merged + 2)
}

/// The literal the chat template emits for an image: one placeholder between
/// the vision sentinels.
///
/// The processor's job is to blow that single `<|image_pad|>` up into one per
/// merged token; [`expand_image_placeholders`] does it at the id level instead
/// of by repeating a 13-character literal a thousand times before tokenizing.
pub const IMAGE_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Expand each single `image_token` in `ids` into `counts[k]` copies.
///
/// The rendered prompt carries one placeholder per image; the model needs one
/// per merged token so the vision features have a row each to land on. Doing
/// this after tokenization keeps the counts exact — no dependence on how the
/// tokenizer would have merged a long repeated literal.
pub fn expand_image_placeholders(
    ids: &[u32],
    image_token: u32,
    counts: &[usize],
) -> Result<Vec<u32>, String> {
    let found = ids.iter().filter(|&&t| t == image_token).count();
    if found != counts.len() {
        return Err(format!(
            "prompt carries {found} image placeholder(s) but {} image(s) were supplied",
            counts.len()
        ));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut k = 0usize;
    for &t in ids {
        if t == image_token {
            if counts[k] == 0 {
                return Err(format!("image {k} expands to zero tokens"));
            }
            out.extend(std::iter::repeat_n(image_token, counts[k]));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

/// Per-token `(t, h, w)` MRoPE positions for a prompt containing image blocks.
///
/// Mirrors `Qwen3VLModel.get_rope_index`. Text tokens advance all three axes
/// together, so a text-only prompt yields `[i, i, i]` — the degenerate case
/// where MRoPE is ordinary 1-D RoPE. An image block instead pins `t` at the
/// block's start and spreads `h`/`w` over the *merged* grid, then the running
/// position resumes at `start + max(grid_h, grid_w)`.
///
/// That last step is why an image costs far fewer positions than tokens: a
/// 32×48 merged grid occupies 1536 token slots but advances the position
/// counter by only 48. Getting it wrong desynchronizes every position after the
/// image — including at decode time, which reads the final counter.
///
/// `grids[i]` is the **merged** `(h, w)` of image `i` (post-merger, i.e. what
/// the tower emits), in prompt order.
pub fn mrope_positions(
    input_ids: &[u32],
    image_token: u32,
    grids: &[(usize, usize)],
) -> Result<Vec<[i32; 3]>, String> {
    let runs = crate::vision_splice::image_token_runs(input_ids, image_token);
    if runs.len() != grids.len() {
        return Err(format!(
            "prompt has {} image-token run(s) but {} grid(s) were supplied",
            runs.len(),
            grids.len()
        ));
    }

    let mut out = Vec::with_capacity(input_ids.len());
    let mut cur: i32 = 0;
    let mut next_run = 0usize;
    let mut i = 0usize;
    while i < input_ids.len() {
        match runs.get(next_run) {
            Some(&(start, len)) if start == i => {
                let (lh, lw) = grids[next_run];
                if lh * lw != len {
                    return Err(format!(
                        "image {next_run} has a {lh}×{lw} merged grid ({} tokens) but the \
                         prompt reserved {len} placeholders",
                        lh * lw
                    ));
                }
                for k in 0..len {
                    out.push([cur, cur + (k / lw) as i32, cur + (k % lw) as i32]);
                }
                // The next text token starts past the image's widest extent,
                // not past its token count.
                cur += lh.max(lw) as i32;
                i += len;
                next_run += 1;
            }
            _ => {
                out.push([cur, cur, cur]);
                cur += 1;
                i += 1;
            }
        }
    }
    Ok(out)
}

/// The position the token *after* this prompt occupies.
///
/// Decode continues on the fused scalar-offset rope, which needs a single
/// number rather than a triple — legitimate because all three axes have
/// realigned by then. This is that number, and it is **not** `prompt.len()`
/// once an image is in play.
pub fn next_position(positions: &[[i32; 3]]) -> i32 {
    positions
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

#[cfg(test)]
mod placeholder_tests {
    use super::expand_image_placeholders;

    const IMG: u32 = 248056;

    #[test]
    fn expands_each_placeholder_to_its_own_count() {
        let ids = [1u32, IMG, 2, IMG, 3];
        let out = expand_image_placeholders(&ids, IMG, &[3, 2]).expect("expand");
        assert_eq!(out, vec![1, IMG, IMG, IMG, 2, IMG, IMG, 3]);
    }

    #[test]
    fn text_only_prompts_pass_through_untouched() {
        let ids = [1u32, 2, 3];
        assert_eq!(expand_image_placeholders(&ids, IMG, &[]).unwrap(), ids);
    }

    #[test]
    fn count_mismatch_is_rejected() {
        let ids = [1u32, IMG, 2];
        assert!(expand_image_placeholders(&ids, IMG, &[]).is_err());
        assert!(expand_image_placeholders(&ids, IMG, &[4, 4]).is_err());
        assert!(expand_image_placeholders(&ids, IMG, &[0]).is_err());
    }
}

#[cfg(test)]
mod position_tests {
    use super::{mrope_positions, next_position};

    const IMG: u32 = 248056;

    #[test]
    fn text_only_prompts_get_identity_positions() {
        let ids = [1u32, 2, 3, 4];
        let pos = mrope_positions(&ids, IMG, &[]).expect("positions");
        assert_eq!(pos, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
        assert_eq!(next_position(&pos), 4);
    }

    /// The image's tokens share `t` and tile `(h, w)` over the merged grid.
    #[test]
    fn image_tokens_tile_the_merged_grid() {
        // 2×3 merged grid → 6 placeholders.
        let ids = [7u32, IMG, IMG, IMG, IMG, IMG, IMG, 9];
        let pos = mrope_positions(&ids, IMG, &[(2, 3)]).expect("positions");
        assert_eq!(pos[0], [0, 0, 0], "leading text");
        // Image starts at position 1.
        assert_eq!(
            &pos[1..7],
            &[
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 2, 1],
                [1, 2, 2],
                [1, 2, 3],
            ]
        );
        // Trailing text resumes at 1 + max(2, 3) = 4, NOT at 1 + 6.
        assert_eq!(pos[7], [4, 4, 4]);
        assert_eq!(next_position(&pos), 5);
    }

    /// A wide image advances the counter by its width; a tall one by its
    /// height. Using the token count instead would desync everything after it.
    #[test]
    fn position_advance_is_the_widest_extent_not_the_token_count() {
        for (grid, advance) in [((2usize, 8usize), 8i32), ((8, 2), 8), ((4, 4), 4)] {
            let n = grid.0 * grid.1;
            let mut ids = vec![IMG; n];
            ids.push(5);
            let pos = mrope_positions(&ids, IMG, &[grid]).expect("positions");
            assert_eq!(
                pos[n],
                [advance, advance, advance],
                "{grid:?} should advance by {advance}, not {n}"
            );
        }
    }

    #[test]
    fn multiple_images_accumulate_independently() {
        let mut ids = vec![1u32];
        ids.extend(std::iter::repeat_n(IMG, 4)); // 2×2
        ids.push(2);
        ids.extend(std::iter::repeat_n(IMG, 6)); // 2×3
        ids.push(3);
        let pos = mrope_positions(&ids, IMG, &[(2, 2), (2, 3)]).expect("positions");
        // text(0) img@1 (advance 2) text@3 img@4 (advance 3) text@7
        assert_eq!(pos[0], [0, 0, 0]);
        assert_eq!(pos[1], [1, 1, 1]);
        assert_eq!(pos[5], [3, 3, 3]);
        assert_eq!(pos[6], [4, 4, 4]);
        assert_eq!(pos[12], [7, 7, 7]);
    }

    #[test]
    fn grid_mismatch_is_rejected() {
        let ids = [IMG, IMG, IMG];
        assert!(mrope_positions(&ids, IMG, &[(2, 2)]).is_err());
        assert!(mrope_positions(&ids, IMG, &[]).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACTOR: usize = 32; // patch 16 × merge 2

    #[test]
    fn smart_resize_lands_on_the_factor_grid() {
        for (h, w) in [(480usize, 640usize), (1000, 1000), (33, 71)] {
            let (rh, rw) = smart_resize(h, w, FACTOR, 4 * 32 * 32, 16384 * 32 * 32).unwrap();
            assert_eq!(rh % FACTOR, 0, "{h}×{w} → {rh}×{rw}");
            assert_eq!(rw % FACTOR, 0, "{h}×{w} → {rh}×{rw}");
        }
    }

    #[test]
    fn smart_resize_shrinks_to_the_pixel_budget() {
        let max_pixels = 256 * 32 * 32;
        let (h, w) = smart_resize(4000, 3000, FACTOR, 4 * 32 * 32, max_pixels).unwrap();
        assert!(h * w <= max_pixels, "{h}×{w} = {} px", h * w);
        // Aspect ratio is preserved to within one factor step on each side.
        let want = 4000.0 / 3000.0;
        let got = h as f64 / w as f64;
        assert!((got - want).abs() < 0.1, "aspect {got} vs {want}");
    }

    #[test]
    fn smart_resize_grows_to_the_minimum() {
        let min_pixels = 64 * 32 * 32;
        let (h, w) = smart_resize(40, 30, FACTOR, min_pixels, 16384 * 32 * 32).unwrap();
        assert!(h * w >= min_pixels, "{h}×{w} = {} px", h * w);
    }

    #[test]
    fn smart_resize_rejects_absurd_aspect_ratios() {
        assert!(smart_resize(1, 5000, FACTOR, 1024, 1_000_000).is_err());
    }

    /// The merge-block row order is the whole reason the merger can be a
    /// reshape. Pin it: patch `(pr, pc)` must land at the row its block
    /// position implies, not at raster index `pr * gw + pc`.
    #[test]
    fn patchify_emits_merge_block_row_order() {
        const PS: usize = 2;
        const MERGE: usize = 2;
        const T: usize = 1;
        // 3 blocks wide, 2 blocks tall → 6×4 patches → 12×8 pixels.
        let (gh, gw) = (4usize, 6usize);
        let (h, w) = (gh * PS, gw * PS);
        // Encode each pixel's patch coordinates in its red channel so a row can
        // be traced back to the patch it came from.
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let (pr, pc) = (y / PS, x / PS);
                rgb[(y * w + x) * 3] = (pr * gw + pc) as u8;
            }
        }
        let out = patchify(&rgb, h, w, PS, T, MERGE);
        let per_patch = T * PS * PS * 3;
        assert_eq!(out.len(), gh * gw * per_patch);

        let decode_red = |row: usize| -> usize {
            let v = out[row * per_patch] * IMAGE_STD[0] + IMAGE_MEAN[0];
            (v * 255.0).round() as usize
        };
        // Expected traversal: block(0,0) patches (0,0),(0,1),(1,0),(1,1);
        // then block(0,1) → patches (0,2),(0,3),(1,2),(1,3); …
        let mut row = 0;
        for br in 0..gh / MERGE {
            for bc in 0..gw / MERGE {
                for ir in 0..MERGE {
                    for ic in 0..MERGE {
                        let (pr, pc) = (br * MERGE + ir, bc * MERGE + ic);
                        assert_eq!(
                            decode_red(row),
                            pr * gw + pc,
                            "row {row} should hold patch ({pr},{pc})"
                        );
                        row += 1;
                    }
                }
            }
        }
    }

    /// Within a row the features run `(t, py, px, c)`. A `(c, t, py, px)` port
    /// — what the torch checkpoint layout would want — reads the same bytes in
    /// a different order and produces confident nonsense.
    #[test]
    fn patchify_packs_channels_innermost_and_duplicates_frames() {
        const PS: usize = 2;
        const T: usize = 2;
        let (h, w) = (PS, PS);
        let mut rgb = vec![0u8; h * w * 3];
        for (i, px) in rgb.chunks_mut(3).enumerate() {
            px[0] = (10 * i + 1) as u8;
            px[1] = (10 * i + 2) as u8;
            px[2] = (10 * i + 3) as u8;
        }
        let out = patchify(&rgb, h, w, PS, T, 1);
        assert_eq!(out.len(), T * PS * PS * 3);

        let denorm = |v: f32, c: usize| (v * IMAGE_STD[c] + IMAGE_MEAN[c]) * 255.0;
        // Frame 0, pixel 0: channels adjacent.
        assert!((denorm(out[0], 0) - 1.0).abs() < 0.5);
        assert!((denorm(out[1], 1) - 2.0).abs() < 0.5);
        assert!((denorm(out[2], 2) - 3.0).abs() < 0.5);
        // Frame 1 repeats frame 0 — a still image padded to the temporal depth.
        let frame = PS * PS * 3;
        for i in 0..frame {
            assert_eq!(out[i], out[frame + i], "temporal frames must match");
        }
    }

    /// Normalization is CLIP-style mean/std, not the bare `[0, 1]` rescale
    /// Gemma 4 uses — a mid-grey pixel must land near zero, not near 0.5.
    #[test]
    fn patchify_applies_clip_normalization() {
        let mid = (IMAGE_MEAN[0] * 255.0).round() as u8;
        let rgb = vec![mid; 3];
        let out = patchify(&rgb, 1, 1, 1, 1, 1);
        assert!(
            out[0].abs() < 0.02,
            "mean-valued pixel should normalize to ~0, got {}",
            out[0]
        );
    }
}
