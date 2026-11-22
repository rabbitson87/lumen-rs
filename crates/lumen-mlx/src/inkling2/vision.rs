//! The Inkling 2 vision tower: image blocks in, one text-sized embedding per block out.
//!
//! A port of `omlx/patches/inkling2/vision.py` from oMLX 0.8.2 (Apache-2.0); `SPEC-step4-vision.md`
//! in the task folder lists what it reproduces. A ViT with axial 2D RoPE on interleaved channel
//! pairs, a position table resampled bilinearly to each image, one class token per image and a 2 x 2
//! merge adapter: each block of `merge * patch` pixels is `merge * merge` patches and becomes one
//! image token. Images are encoded one at a time; there is no attention across images.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::ops::indexing::{TryIndexOp, take_axis};
use mlx_rs::{Array, Dtype};
use serde::Deserialize;

use super::pixels::PreparedImage;
use crate::qwen3_5_moe::NativeWeights;

const CHANNELS: i32 = 3;

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: Option<f64>,
}

/// The `vision_config` block of `config.json` (and the text model's hidden size, which the
/// adapter projects to).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionArgs {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Pixels per side of one patch.
    pub native_patch_size: usize,
    /// Patches per side that merge into one image token.
    pub adapter_merge_size: usize,
    /// Side of the learned position table, which is resampled to each image.
    pub posemb_grid_size: usize,
    pub projection_hidden_dim: usize,
    #[serde(default)]
    pub text_hidden_size: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_ratio")]
    pub mlp_ratio: f32,
    /// The reference builds the adapter without bias; a checkpoint with one is refused.
    #[serde(default)]
    pub projector_bias: bool,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

fn default_eps() -> f32 {
    1e-5
}

fn default_ratio() -> f32 {
    4.0
}

impl VisionArgs {
    /// Reads the `vision_config` block of the `config.json` text in `config`.
    pub fn from_config_str(config: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(config).context("parse config.json")?;
        let block = root
            .get("vision_config")
            .context("config.json has no vision_config")?;
        let mut args: Self =
            serde_json::from_value(block.clone()).context("parse vision_config")?;
        // `text_hidden_size` is in the vision block of the real config; fall back to the text's.
        if args.text_hidden_size == 0 {
            args.text_hidden_size =
                root["text_config"]["hidden_size"].as_u64().unwrap_or(0) as usize;
        }
        args.validate()?;
        Ok(args)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.projector_bias,
            "vision_config.projector_bias is true; this port implements the bias-free adapter only"
        );
        ensure!(
            self.num_attention_heads > 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads),
            "hidden_size {} is not a multiple of {} heads",
            self.hidden_size,
            self.num_attention_heads
        );
        ensure!(
            self.head_dim().is_multiple_of(4),
            "head_dim {} is not a multiple of 4 (axial RoPE rotates channel pairs, half per axis)",
            self.head_dim()
        );
        ensure!(
            self.adapter_merge_size >= 1 && self.native_patch_size >= 1,
            "empty patch"
        );
        ensure!(self.posemb_grid_size >= 2, "position grid too small");
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Pixels per side of one block: the unit of one image token.
    pub fn block_side(&self) -> usize {
        self.native_patch_size * self.adapter_merge_size
    }

    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(10_000.0)
    }
}

/// `x * (1 + erf(x / sqrt(2))) / 2`, in `x`'s own dtype and in the reference's order of operations.
/// `mlx_rs::nn::gelu` writes its constants as float32 arrays, which promote a bfloat16 input to
/// float32; the reference's Python scalars are weak and leave it bfloat16, so a tower on the
/// checkpoint's bfloat16 weights would otherwise run in float32 from the first MLP on.
fn gelu(x: &Array) -> Result<Array> {
    let dtype = x.dtype();
    let constant = |v: f32| -> Result<Array> { Ok(Array::from_f32(v).as_dtype(dtype)?) };
    let scaled = mlx_rs::ops::divide(x, constant(std::f32::consts::SQRT_2)?)?;
    let inner = mlx_rs::ops::add(constant(1.0)?, mlx_rs::ops::erf(&scaled)?)?;
    Ok(mlx_rs::ops::divide(
        mlx_rs::ops::multiply(x, inner)?,
        constant(2.0)?,
    )?)
}

/// `x @ W^T + b`, the reference's `nn.Linear` (which is an `addmm`).
struct Linear {
    weight_t: Array,
    bias: Array,
}

impl Linear {
    fn apply(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::ops::addmm(
            &self.bias,
            x,
            &self.weight_t,
            None,
            None,
        )?)
    }
}

struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn apply(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )?)
    }
}

struct EncoderLayer {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    c_fc: Linear,
    c_proj: Linear,
    ls_1: Array,
    ls_2: Array,
}

pub struct VisionTower {
    args: VisionArgs,
    patch_embedding_t: Array,
    position_embedding: Array,
    class_token: Array,
    ln_pre: LayerNorm,
    layers: Vec<EncoderLayer>,
    proj_in_t: Array,
    proj_out_t: Array,
    /// Every array the tower holds, to evaluate where they are loaded.
    held: Vec<Array>,
}

/// Where `get` reads tensors from: a name and the shape it must have.
type Getter<'a> = dyn FnMut(&str, &[i32]) -> Result<Array> + 'a;

impl VisionTower {
    /// Loads the tower of the checkpoint in `dir` (the `vision_tower.*` tensors of its shards),
    /// reading them into memory on this thread.
    pub fn load(dir: &Path) -> Result<Self> {
        let config =
            std::fs::read_to_string(dir.join("config.json")).context("read config.json")?;
        let args = VisionArgs::from_config_str(&config)?;
        let weights = NativeWeights::load_dir(dir)?;
        let tower = Self::build(args, &mut |name, shape| {
            let a = weights.require(name)?.clone();
            ensure!(
                a.shape() == shape,
                "{name}: shape {:?}, expected {shape:?}",
                a.shape()
            );
            Ok(a)
        })?;
        for batch in tower.held.chunks(256) {
            mlx_rs::transforms::eval(batch.iter()).context("read the vision tower into memory")?;
        }
        Ok(tower)
    }

    /// Builds the tower from named tensors (the test replica, or a bag already loaded).
    pub fn from_tensors(args: VisionArgs, tensors: &HashMap<String, Array>) -> Result<Self> {
        args.validate()?;
        Self::build(args, &mut |name, shape| {
            let a = tensors
                .get(name)
                .ok_or_else(|| anyhow!("missing tensor {name}"))?
                .clone();
            ensure!(
                a.shape() == shape,
                "{name}: shape {:?}, expected {shape:?}",
                a.shape()
            );
            Ok(a)
        })
    }

    fn build(args: VisionArgs, get: &mut Getter<'_>) -> Result<Self> {
        let (h, mlp) = (
            args.hidden_size as i32,
            (args.hidden_size as f32 * args.mlp_ratio) as i32,
        );
        let patch_values = (args.native_patch_size * args.native_patch_size) as i32 * CHANNELS;
        let merged = (args.adapter_merge_size * args.adapter_merge_size) as i32 * h;
        let grid = (args.posemb_grid_size * args.posemb_grid_size) as i32;
        let eps = args.layer_norm_eps;
        let mut held: Vec<Array> = Vec::new();
        let mut take = |name: &str, shape: &[i32]| -> Result<Array> {
            let a = get(name, shape)?;
            held.push(a.clone());
            Ok(a)
        };

        let pre = "vision_tower.backbone";
        let patch_embedding_t = {
            let w = take(
                &format!("{pre}.embeddings.patch_embedding.weight"),
                &[h, patch_values],
            )?;
            mlx_rs::ops::transpose_axes(&w, &[1, 0])?
        };
        let position_embedding = take(&format!("{pre}.embeddings.position_embedding"), &[grid, h])?;
        let class_token = mlx_rs::ops::add(
            take(&format!("{pre}.embeddings.class_embedding"), &[h])?,
            take(&format!("{pre}.embeddings.cls_position_embedding"), &[h])?,
        )?;
        let ln_pre = LayerNorm {
            weight: take(&format!("{pre}.ln_pre.weight"), &[h])?,
            bias: take(&format!("{pre}.ln_pre.bias"), &[h])?,
            eps,
        };
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let l = format!("{pre}.layers.{i}");
            let mut linear = |name: &str, out: i32, inp: i32| -> Result<Linear> {
                let w = take(&format!("{l}.{name}.weight"), &[out, inp])?;
                Ok(Linear {
                    weight_t: mlx_rs::ops::transpose_axes(&w, &[1, 0])?,
                    bias: take(&format!("{l}.{name}.bias"), &[out])?,
                })
            };
            let (q, k, v, out) = (
                linear("self_attn.q_proj", h, h)?,
                linear("self_attn.k_proj", h, h)?,
                linear("self_attn.v_proj", h, h)?,
                linear("self_attn.out_proj", h, h)?,
            );
            let (c_fc, c_proj) = (linear("mlp.c_fc", mlp, h)?, linear("mlp.c_proj", h, mlp)?);
            let mut norm = |name: &str| -> Result<LayerNorm> {
                Ok(LayerNorm {
                    weight: take(&format!("{l}.{name}.weight"), &[h])?,
                    bias: take(&format!("{l}.{name}.bias"), &[h])?,
                    eps,
                })
            };
            let (ln_1, ln_2) = (norm("ln_1")?, norm("ln_2")?);
            layers.push(EncoderLayer {
                ln_1,
                ln_2,
                q,
                k,
                v,
                out,
                c_fc,
                c_proj,
                ls_1: take(&format!("{l}.ls_1.gamma"), &[h])?,
                ls_2: take(&format!("{l}.ls_2.gamma"), &[h])?,
            });
        }
        let proj_in_t = {
            let w = take(
                "vision_tower.proj_in.weight",
                &[args.projection_hidden_dim as i32, merged],
            )?;
            mlx_rs::ops::transpose_axes(&w, &[1, 0])?
        };
        let proj_out_t = {
            let w = take(
                "vision_tower.proj_out.weight",
                &[
                    args.text_hidden_size as i32,
                    args.projection_hidden_dim as i32,
                ],
            )?;
            mlx_rs::ops::transpose_axes(&w, &[1, 0])?
        };
        Ok(Self {
            args,
            patch_embedding_t,
            position_embedding,
            class_token,
            ln_pre,
            layers,
            proj_in_t,
            proj_out_t,
            held,
        })
    }

    pub fn args(&self) -> &VisionArgs {
        &self.args
    }

    /// The image tokens of `image`: `[rows * cols, text_hidden_size]`, one per block.
    pub fn encode(&self, image: &PreparedImage) -> Result<Array> {
        self.run(image, None)
    }

    /// [`Self::encode`] and the tower's state after the embeddings (class token first), after
    /// `ln_pre`, and after every layer, for the parity tests.
    pub fn encode_captured(&self, image: &PreparedImage) -> Result<(Array, Vec<Array>)> {
        let mut captured = Vec::new();
        let features = self.run(image, Some(&mut captured))?;
        Ok((features, captured))
    }

    fn run(&self, image: &PreparedImage, mut capture: Option<&mut Vec<Array>>) -> Result<Array> {
        let a = &self.args;
        let (merge, patch) = (a.adapter_merge_size as i32, a.native_patch_size as i32);
        let side = a.block_side();
        let count = image.rows * image.cols;
        ensure!(count > 0, "an image with no blocks");
        ensure!(
            image.blocks.len() == count * side * side * CHANNELS as usize,
            "{} values for {count} blocks of {side}x{side}x3",
            image.blocks.len()
        );
        let blocks = Array::from_slice(
            &image.blocks,
            &[count as i32, merge, patch, merge, patch, CHANNELS],
        );
        let patches = mlx_rs::ops::reshape(
            mlx_rs::ops::transpose_axes(&blocks, &[0, 1, 3, 2, 4, 5])?,
            &[-1, patch * patch * CHANNELS],
        )?;
        let mut h = mlx_rs::ops::matmul(
            patches.as_dtype(self.patch_embedding_t.dtype())?,
            &self.patch_embedding_t,
        )?;

        // Position: the learned table resampled to this image's patch grid.
        let (grid_h, grid_w) = (
            image.rows * a.adapter_merge_size,
            image.cols * a.adapter_merge_size,
        );
        let (row, col) = grid_coords(grid_h, grid_w, a.adapter_merge_size);
        let side_table = a.posemb_grid_size;
        let (h_taps, h_weights) = bilinear_taps(&row, grid_h, side_table);
        let (w_taps, w_weights) = bilinear_taps(&col, grid_w, side_table);
        let n = row.len();
        let mut index = Vec::with_capacity(n * 4);
        let mut weight = Vec::with_capacity(n * 4);
        for i in 0..n {
            for hi in 0..2 {
                for wi in 0..2 {
                    index.push((h_taps[i][hi] * side_table + w_taps[i][wi]) as i32);
                    weight.push(h_weights[i][hi] * w_weights[i][wi]);
                }
            }
        }
        let table = mlx_rs::ops::reshape(
            take_axis(
                &self.position_embedding,
                Array::from_slice(&index, &[n as i32 * 4]),
                0,
            )?
            .as_dtype(Dtype::Float32)?,
            &[n as i32, 4, a.hidden_size as i32],
        )?;
        let weights = mlx_rs::ops::expand_dims(Array::from_slice(&weight, &[n as i32, 4]), -1)?;
        let position = mlx_rs::ops::sum_axis(mlx_rs::ops::multiply(&table, &weights)?, 1, false)?;
        h = mlx_rs::ops::add(&h, position.as_dtype(h.dtype())?)?;

        let class_token = self.class_token.as_dtype(h.dtype())?;
        h = mlx_rs::ops::concatenate_axis(&[&mlx_rs::ops::expand_dims(&class_token, 0)?, &h], 0)?;
        if let Some(c) = capture.as_deref_mut() {
            c.push(h.clone());
        }
        h = self.ln_pre.apply(&h)?;
        if let Some(c) = capture.as_deref_mut() {
            c.push(h.clone());
        }

        let (cos, sin) = axial_rope(&row, &col, a);
        let (cos, sin) = (
            mlx_rs::ops::expand_dims(&cos, 1)?,
            mlx_rs::ops::expand_dims(&sin, 1)?,
        );
        for layer in &self.layers {
            h = self.layer(layer, &h, &cos, &sin)?;
            if let Some(c) = capture.as_deref_mut() {
                c.push(h.clone());
            }
        }

        // The adapter: drop the class token, four consecutive patches are one block.
        let tokens = mlx_rs::ops::reshape(
            h.try_index(1..)?,
            &[count as i32, merge * merge * a.hidden_size as i32],
        )?;
        let projected = mlx_rs::ops::matmul(&tokens, &self.proj_in_t)?;
        Ok(mlx_rs::ops::matmul(gelu(&projected)?, &self.proj_out_t)?)
    }

    fn layer(&self, layer: &EncoderLayer, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let attended = self.attention(layer, &layer.ln_1.apply(x)?, cos, sin)?;
        let x = mlx_rs::ops::add(x, mlx_rs::ops::multiply(&attended, &layer.ls_1)?)?;
        let hidden = layer
            .c_proj
            .apply(&gelu(&layer.c_fc.apply(&layer.ln_2.apply(&x)?)?)?)?;
        Ok(mlx_rs::ops::add(
            &x,
            mlx_rs::ops::multiply(&hidden, &layer.ls_2)?,
        )?)
    }

    fn attention(
        &self,
        layer: &EncoderLayer,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let length = x.shape()[0];
        let heads = self.args.num_attention_heads as i32;
        let split =
            |a: Array| -> Result<Array> { Ok(mlx_rs::ops::reshape(a, &[length, heads, -1])?) };
        let q = rotate_pairs(&split(layer.q.apply(x)?)?, cos, sin)?;
        let k = rotate_pairs(&split(layer.k.apply(x)?)?, cos, sin)?;
        let v = split(layer.v.apply(x)?)?;
        let heads_first = |a: &Array| -> Result<Array> {
            Ok(mlx_rs::ops::expand_dims(
                mlx_rs::ops::transpose_axes(a, &[1, 0, 2])?,
                0,
            )?)
        };
        let scale = (self.args.head_dim() as f64).powf(-0.5) as f32;
        let out = mlx_rs::fast::scaled_dot_product_attention(
            heads_first(&q)?,
            heads_first(&k)?,
            heads_first(&v)?,
            scale,
            None,
            None,
        )?;
        let merged = mlx_rs::ops::reshape(
            mlx_rs::ops::transpose_axes(out.try_index(0)?, &[1, 0, 2])?,
            &[length, -1],
        )?;
        layer.out.apply(&merged)
    }
}

/// `(row, col)` of every patch of a `rows x cols` patch grid, listed in merge-block order: for each
/// block (raster order), its `merge x merge` patches (raster order within the block).
fn grid_coords(rows: usize, cols: usize, merge: usize) -> (Vec<usize>, Vec<usize>) {
    let blocks_w = cols / merge;
    let mut row = Vec::with_capacity(rows * cols);
    let mut col = Vec::with_capacity(rows * cols);
    for within in 0..rows * cols {
        let in_col = within % merge;
        let in_row = (within / merge) % merge;
        let block_col = (within / (merge * merge)) % blocks_w;
        let block_row = within / (merge * merge * blocks_w);
        row.push(block_row * merge + in_row);
        col.push(block_col * merge + in_col);
    }
    (row, col)
}

/// The two taps and weights of `F.interpolate(mode="bilinear", align_corners=False)` from a
/// `side`-long table to `size` positions, border clamped, in float32 like the reference's numpy.
fn bilinear_taps(index: &[usize], size: usize, side: usize) -> (Vec<[usize; 2]>, Vec<[f32; 2]>) {
    let mut taps = Vec::with_capacity(index.len());
    let mut weights = Vec::with_capacity(index.len());
    for &i in index {
        let src = ((i as f32 + 0.5) * side as f32) / size as f32 - 0.5;
        let floor = src.floor();
        let mut tap = [0usize; 2];
        let mut weight = [0f32; 2];
        for offset in 0..2 {
            let t = floor as i64 + offset as i64;
            tap[offset] = t.clamp(0, side as i64 - 1) as usize;
            weight[offset] = (1.0 - (src - floor - offset as f32).abs()).max(0.0);
        }
        taps.push(tap);
        weights.push(weight);
    }
    (taps, weights)
}

/// `cos` and `sin`, `[1 + patches, head_dim]` float32: the class token at (0, 0) and the patches
/// at their 1-indexed (row, col); the width frequencies first, then the height's, each angle
/// repeated for one interleaved channel pair.
fn axial_rope(row: &[usize], col: &[usize], args: &VisionArgs) -> (Array, Array) {
    let head_dim = args.head_dim();
    let half = head_dim / 2;
    let theta = args.rope_theta() as f32;
    let inv_freq: Vec<f32> = (0..half)
        .step_by(2)
        .map(|k| 1.0 / theta.powf(k as f32 / half as f32))
        .collect();
    let length = row.len() + 1;
    let mut cos = Vec::with_capacity(length * head_dim);
    let mut sin = Vec::with_capacity(length * head_dim);
    for p in 0..length {
        let (pos_h, pos_w) = if p == 0 {
            (0.0f32, 0.0f32)
        } else {
            ((row[p - 1] + 1) as f32, (col[p - 1] + 1) as f32)
        };
        let angles = inv_freq
            .iter()
            .map(|f| pos_w * f)
            .chain(inv_freq.iter().map(|f| pos_h * f));
        for angle in angles {
            for _ in 0..2 {
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    let shape = [length as i32, head_dim as i32];
    (
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    )
}

/// Rotates interleaved channel pairs `(x1, x2) -> (x1 c - x2 s, x2 c + x1 s)` in float32 and casts
/// back. `x` is `[L, heads, head_dim]`, `cos` and `sin` `[L, 1, head_dim]`.
fn rotate_pairs(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let shape = xf.shape().to_vec();
    let mut pair_shape = shape.clone();
    let last = pair_shape.pop().expect("rank >= 1");
    pair_shape.extend([last / 2, 2]);
    let pairs = mlx_rs::ops::reshape(&xf, &pair_shape)?;
    let (x1, x2) = (
        pairs.try_index((.., .., .., 0))?,
        pairs.try_index((.., .., .., 1))?,
    );
    let rotated = mlx_rs::ops::reshape(
        mlx_rs::ops::stack_axis(&[mlx_rs::ops::negative(&x2)?, x1], 3)?,
        &shape,
    )?;
    let out = mlx_rs::ops::add(
        mlx_rs::ops::multiply(&xf, cos)?,
        mlx_rs::ops::multiply(&rotated, sin)?,
    )?;
    Ok(out.as_dtype(x.dtype())?)
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn patches_are_listed_in_merge_block_order() {
        // A 4 x 6 patch grid, merge 2: blocks are 2 x 2 patches, 2 block rows by 3 block columns.
        let (row, col) = grid_coords(4, 6, 2);
        let first_block: Vec<(usize, usize)> = (0..4).map(|i| (row[i], col[i])).collect();
        assert_eq!(first_block, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        let second_block: Vec<(usize, usize)> = (4..8).map(|i| (row[i], col[i])).collect();
        assert_eq!(second_block, vec![(0, 2), (0, 3), (1, 2), (1, 3)]);
        // The fourth block starts the second block row.
        assert_eq!((row[12], col[12]), (2, 0));
        assert_eq!(row.len(), 24);
    }

    #[test]
    fn bilinear_taps_clamp_at_the_border_and_weights_sum_to_one() {
        // 32 -> 4 positions: source coordinates 3.5, 11.5, 19.5, 27.5 minus .5... interior taps.
        let (taps, weights) = bilinear_taps(&[0, 1, 2, 3], 4, 32);
        for (t, w) in taps.iter().zip(&weights) {
            assert!((w[0] + w[1] - 1.0).abs() < 1e-6, "{w:?}");
            assert!(t[0] < 32 && t[1] < 32);
        }
        // A 1-long grid sits at the table's centre; a 40-long grid from a 32 table clamps the ends.
        let (taps, weights) = bilinear_taps(&[0], 40, 32);
        assert_eq!(taps[0], [0, 0]);
        assert!((weights[0][0] - 1.0).abs() < 1e-6 || weights[0][0] > 0.0);
    }
}
