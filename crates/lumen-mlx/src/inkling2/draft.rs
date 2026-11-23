//! The Inkling 2 DSpark draft head (`mtp.safetensors`): a small block drafter that proposes the
//! next tokens for the target to check.
//!
//! A port of `omlx/patches/inkling2/dspark.py` and of the row-wise draft forward of
//! `omlx/speculative/dflash/drafter.py` (`_draft_batched`), oMLX 0.8.2 (Apache-2.0), for one
//! sequence; `SPEC-step5-mtp.md` in the task folder lists what it reproduces. Each layer mixes
//! the target's context taps (the stream entering some layer visits and the final-norm output)
//! with its own softmax weights, projects them to context keys and values, and attends to the
//! newest `window - 1` of them and to its own block, in both directions; the block's mask
//! positions propose through the target's LM head plus a recurrent Markov head.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlx_rs::ops::indexing::{TryIndexOp, take_axis};
use mlx_rs::{Array, Dtype};
use serde::Deserialize;

use super::model::Inkling2Model;

/// The file the checkpoint keeps the draft in.
pub const DRAFT_FILE: &str = "mtp.safetensors";

/// Rows of the LM head folded into the Markov head at a time.
const FOLD_CHUNK: i32 = 16_384;

/// The `mtp_config` block of `config.json`, with the few values the draft takes from the target.
#[derive(Debug, Clone)]
pub struct DraftArgs {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Context rows each layer sees: `sliding_window_size - 1`.
    pub context_rows: usize,
    /// The anchor plus the mask positions; `block_size - 1` tokens are proposed.
    pub block_size: usize,
    pub markov_rank: usize,
    /// The target's layer visits (and, for the visit count itself, its final norm) the draft reads.
    pub target_layer_ids: Vec<usize>,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f32,
}

#[derive(Debug, Deserialize)]
struct MtpBlock {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    rope_theta: f64,
    sliding_window_size: usize,
    block_size: usize,
    markov_rank: usize,
    #[serde(default)]
    markov_head_type: String,
    target_layer_ids: Vec<usize>,
}

fn default_eps() -> f32 {
    1e-6
}

fn default_theta() -> f64 {
    1e6
}

impl DraftArgs {
    /// The draft the checkpoint's `config.json` declares, for `target`. `None` when it declares
    /// none (no `mtp_config`).
    pub fn from_config_str(config: &str, target: &super::TextArgs) -> Result<Option<Self>> {
        let root: serde_json::Value = serde_json::from_str(config).context("parse config.json")?;
        let Some(block) = root.get("mtp_config").filter(|b| !b.is_null()) else {
            return Ok(None);
        };
        let b: MtpBlock = serde_json::from_value(block.clone()).context("parse mtp_config")?;
        ensure!(
            !b.q_bias && !b.o_bias,
            "mtp_config has projection biases; this port implements the bias-free draft only"
        );
        ensure!(
            b.markov_head_type == "rnn_hidden",
            "mtp_config.markov_head_type is {:?}, this port implements \"rnn_hidden\"",
            b.markov_head_type
        );
        ensure!(
            b.hidden_size == target.hidden_size,
            "the draft is {} wide, the target {}",
            b.hidden_size,
            target.hidden_size
        );
        ensure!(
            b.num_key_value_heads > 0
                && b.num_attention_heads.is_multiple_of(b.num_key_value_heads),
            "{} query heads over {} key/value heads",
            b.num_attention_heads,
            b.num_key_value_heads
        );
        ensure!(
            b.head_dim.is_multiple_of(2),
            "head_dim {} is odd",
            b.head_dim
        );
        ensure!(
            b.sliding_window_size >= 2,
            "a window of {}",
            b.sliding_window_size
        );
        ensure!(b.block_size >= 2, "a block of {}", b.block_size);
        ensure!(!b.target_layer_ids.is_empty(), "no target layer ids");
        Ok(Some(Self {
            hidden_size: b.hidden_size,
            num_layers: b.num_hidden_layers,
            num_heads: b.num_attention_heads,
            num_kv_heads: b.num_key_value_heads,
            head_dim: b.head_dim,
            intermediate_size: b.intermediate_size,
            rms_norm_eps: b.rms_norm_eps,
            rope_theta: b.rope_theta as f32,
            context_rows: b.sliding_window_size - 1,
            block_size: b.block_size,
            markov_rank: b.markov_rank,
            target_layer_ids: b.target_layer_ids,
            vocab_size: target.vocab_size,
            unpadded_vocab_size: target.unpadded_vocab_size,
            final_logit_softcapping: target.final_logit_softcapping,
        }))
    }

    /// Tokens one block proposes.
    pub fn drafts_per_block(&self) -> usize {
        self.block_size - 1
    }

    fn taps(&self) -> usize {
        self.target_layer_ids.len()
    }
}

/// `y = x @ W^T` for a dense `[out, in]` weight, kept transposed.
struct Dense(Array);

impl Dense {
    fn new(weight: Array) -> Result<Self> {
        Ok(Self(mlx_rs::ops::transpose_axes(&weight, &[1, 0])?))
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::ops::matmul(x, &self.0)?)
    }
}

struct DraftLayer {
    input_norm: Array,
    post_attention_norm: Array,
    /// The norm of this layer's mix of the context taps.
    context_norm: Array,
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    k_ctx: Dense,
    v_ctx: Dense,
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    gate: Dense,
    up: Dense,
    down: Dense,
}

struct MarkovHead {
    /// `[vocab, rank]`.
    w1: Array,
    rnn_proj: Dense,
    rnn_bias: Array,
    context_norm: Array,
    /// `lm_head @ hidden_out` and `lm_head @ hidden_out_bias`, folded once: `[vocab, rank]`, `[vocab]`.
    fold_weight_t: Array,
    fold_bias: Array,
    /// `-inf` past the unpadded vocabulary.
    vocab_bias: Option<Array>,
}

pub struct DraftModel {
    args: DraftArgs,
    layers: Vec<DraftLayer>,
    norm: Array,
    mask_embed: Array,
    /// One softmax row per layer and one for the Markov head: `[layers + 1][taps]`.
    alpha: Vec<Array>,
    markov: MarkovHead,
}

/// What one draft cycle computes, for the parity tests; the tokens are what the backend uses.
pub struct DraftOut {
    pub tokens: Vec<u32>,
    /// `[1, drafts, H]`: the block's normalized output past the anchor.
    pub hidden: Array,
    /// `[1, drafts, vocab]`: the target head's output for it.
    pub base_logits: Array,
    /// `[1, drafts, vocab]`: after the Markov head, the softcap and the vocabulary mask.
    pub step_logits: Array,
}

/// The newest rows of the target's taps and the end position they stop at: what a draft context
/// is rebuilt from, and what a retained conversation keeps beside its caches.
pub struct TapRing {
    rows: Option<Array>,
    end: usize,
}

impl TapRing {
    pub fn new() -> Self {
        Self { rows: None, end: 0 }
    }

    /// Positions held (`end - len()..end`).
    pub fn len(&self) -> usize {
        self.rows.as_ref().map_or(0, |r| r.shape()[1] as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The position after the newest row.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Appends the taps of the next `rows.shape()[1]` positions, keeping the newest `keep`.
    pub fn push(&mut self, rows: &Array, keep: usize) -> Result<()> {
        let n = rows.shape()[1] as usize;
        let joined = match &self.rows {
            Some(old) => mlx_rs::ops::concatenate_axis(&[old, rows], 1)?,
            None => rows.clone(),
        };
        self.rows = Some(newest(&joined, keep)?);
        self.end += n;
        Ok(())
    }

    /// Drops the positions from `len` on (a conversation rewound to `len` tokens).
    pub fn truncate_to(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.end,
            "a tap ring ending at {} cannot be cut to {len}",
            self.end
        );
        let drop = self.end - len;
        if let Some(rows) = &self.rows {
            let held = rows.shape()[1] as usize;
            self.rows = if drop >= held {
                None
            } else {
                Some(owned_slice(rows, 0, (held - drop) as i32)?)
            };
        }
        self.end = len;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn rows_for_test(&self) -> Option<&Array> {
        self.rows.as_ref()
    }

    /// Evaluates the held rows, so the ring does not carry the graph that made them.
    pub fn eval(&self) -> Result<()> {
        if let Some(rows) = &self.rows {
            rows.eval()?;
        }
        Ok(())
    }
}

impl Default for TapRing {
    fn default() -> Self {
        Self::new()
    }
}

/// The rows `from..to` of axis 1, copied so the result does not pin its source.
fn owned_slice(x: &Array, from: i32, to: i32) -> Result<Array> {
    super::ops::owned(&x.try_index((.., from..to, ..))?)
}

/// The newest `keep` rows of axis 1.
fn newest(x: &Array, keep: usize) -> Result<Array> {
    let len = x.shape()[1] as usize;
    if len <= keep {
        return Ok(x.clone());
    }
    owned_slice(x, (len - keep) as i32, len as i32)
}

/// One draft's view of its sequence: the context keys and values each layer attends to, RoPE
/// applied, the newest `context_rows` positions.
pub struct DraftContext {
    /// Per layer, `[1, kv_heads, rows, head_dim]` keys and values.
    rings: Vec<Option<(Array, Array)>>,
    /// Positions that have context rows; the anchor's position.
    total: usize,
    /// The newest raw tap row, `[1, 1, taps * H]`.
    last_row: Option<Array>,
}

impl DraftContext {
    pub fn total(&self) -> usize {
        self.total
    }

    /// The keys and values layer `l` attends to, `[1, kv_heads, rows, head_dim]` each.
    #[cfg(test)]
    pub(super) fn ring(&self, l: usize) -> Option<(&Array, &Array)> {
        self.rings[l].as_ref().map(|(k, v)| (k, v))
    }
}

/// `mix_rmsnorm`: the softmax-weighted sum of the `taps` blocks of `rows` (`[1, n, taps * H]`),
/// RMS-normalized, in the dtype of `rows`.
fn mix_rmsnorm(rows: &Array, alpha: &Array, norm: &Array, taps: i32, eps: f32) -> Result<Array> {
    let (n, width) = (rows.shape()[1], rows.shape()[2]);
    let x = mlx_rs::ops::reshape(rows.as_dtype(Dtype::Float32)?, &[1, n, taps, width / taps])?;
    let a = mlx_rs::ops::reshape(alpha.as_dtype(Dtype::Float32)?, &[taps, 1])?;
    let mixed = mlx_rs::ops::sum_axis(mlx_rs::ops::multiply(&x, &a)?, 2, false)?;
    Ok(
        mlx_rs::fast::rms_norm(&mixed, norm.as_dtype(Dtype::Float32)?, eps)?
            .as_dtype(rows.dtype())?,
    )
}

/// `[1, n, heads * head_dim]` to `[1, heads, n, head_dim]` after an optional norm over `head_dim`.
fn split_heads(x: &Array, heads: i32, head_dim: i32, norm: Option<(&Array, f32)>) -> Result<Array> {
    let n = x.shape()[1];
    let mut y = mlx_rs::ops::reshape(x, &[1, n, heads, head_dim])?;
    if let Some((weight, eps)) = norm {
        y = mlx_rs::fast::rms_norm(&y, weight, eps)?;
    }
    Ok(mlx_rs::ops::transpose_axes(&y, &[0, 2, 1, 3])?)
}

impl DraftModel {
    /// Reads the draft of the checkpoint in `dir` (its `mtp.safetensors`), bound to `target`
    /// (whose embedding and LM head it shares). `None` when the checkpoint declares no draft.
    pub fn load(dir: &Path, target: &Inkling2Model) -> Result<Option<Self>> {
        let config =
            std::fs::read_to_string(dir.join("config.json")).context("read config.json")?;
        let Some(args) = DraftArgs::from_config_str(&config, target.args())? else {
            return Ok(None);
        };
        let path = dir.join(DRAFT_FILE);
        ensure!(
            path.is_file(),
            "config.json declares a draft but {} is missing",
            path.display()
        );
        crate::qwen3_5_moe::validate_safetensors_complete(&path)?;
        let tensors = Array::load_safetensors(&path)
            .map_err(|e| anyhow::anyhow!("load_safetensors({}) failed: {e}", path.display()))?;
        Self::from_tensors(args, tensors, target).map(Some)
    }

    /// Builds the draft from named tensors (the checkpoint's, or the test replica's).
    pub fn from_tensors(
        args: DraftArgs,
        tensors: std::collections::HashMap<String, Array>,
        target: &Inkling2Model,
    ) -> Result<Self> {
        let (h, q_dim, kv_dim, inter, rank) = (
            args.hidden_size as i32,
            (args.num_heads * args.head_dim) as i32,
            (args.num_kv_heads * args.head_dim) as i32,
            args.intermediate_size as i32,
            args.markov_rank as i32,
        );
        let hd = args.head_dim as i32;
        let get = |name: &str, shape: &[i32]| -> Result<Array> {
            let a = tensors
                .get(name)
                .with_context(|| format!("the draft has no tensor {name}"))?
                .clone();
            ensure!(
                a.shape() == shape,
                "{name}: shape {:?}, expected {shape:?}",
                a.shape()
            );
            Ok(a)
        };
        let mut held: Vec<Array> = Vec::new();
        let mut take = |name: &str, shape: &[i32]| -> Result<Array> {
            let a = get(name, shape)?;
            held.push(a.clone());
            Ok(a)
        };
        let taps = args.taps() as i32;
        let layers_plus = args.num_layers as i32 + 1;
        let fuse = take("fuse_weight", &[layers_plus, taps])?;
        let alpha_all = mlx_rs::ops::softmax_axis(fuse.as_dtype(Dtype::Float32)?, -1, None)?
            .as_dtype(Dtype::Bfloat16)?;
        let alpha: Vec<Array> = (0..layers_plus)
            .map(|i| Ok(alpha_all.try_index(i)?))
            .collect::<Result<_>>()?;
        let mut extra: Vec<Array> = alpha.clone();

        let mut layers = Vec::with_capacity(args.num_layers);
        for l in 0..args.num_layers {
            let p = format!("layers.{l}");
            let mut dense = |name: &str, out: i32, inp: i32| -> Result<Dense> {
                Dense::new(take(&format!("{p}.{name}.weight"), &[out, inp])?)
            };
            let (q, k, v, o) = (
                dense("self_attn.q_proj", q_dim, h)?,
                dense("self_attn.k_proj", kv_dim, h)?,
                dense("self_attn.v_proj", kv_dim, h)?,
                dense("self_attn.o_proj", h, q_dim)?,
            );
            let (k_ctx, v_ctx) = (
                dense("self_attn.k_proj_ctx", kv_dim, h)?,
                dense("self_attn.v_proj_ctx", kv_dim, h)?,
            );
            let (gate, up, down) = (
                dense("mlp.gate_proj", inter, h)?,
                dense("mlp.up_proj", inter, h)?,
                dense("mlp.down_proj", h, inter)?,
            );
            layers.push(DraftLayer {
                input_norm: take(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_norm: take(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                context_norm: take(&format!("{p}.dflare_norm.weight"), &[h])?,
                q,
                k,
                v,
                o,
                k_ctx,
                v_ctx,
                q_norm: take(&format!("{p}.self_attn.q_norm.weight"), &[hd])?,
                k_norm: take(&format!("{p}.self_attn.k_norm.weight"), &[hd])?,
                k_norm_ctx: take(&format!("{p}.self_attn.k_norm_ctx.weight"), &[hd])?,
                gate,
                up,
                down,
            });
        }
        let norm = take("norm.weight", &[h])?;
        let mask_embed = take("mask_embed", &[h])?;

        let vocab = args.vocab_size as i32;
        let w1 = take("markov_head.markov_w1.weight", &[vocab, rank])?;
        let rnn_proj = Dense::new(take(
            "markov_head.rnn_proj.weight",
            &[3 * rank, 2 * rank + 2 * h],
        )?)?;
        let rnn_bias = take("markov_head.rnn_proj.bias", &[3 * rank])?;
        let hidden_out_w = take("markov_head.hidden_out.weight", &[h, rank])?;
        let hidden_out_b = take("markov_head.hidden_out.bias", &[h])?;
        let context_norm = take("markov_head.dflare_norm.weight", &[h])?;

        // The target's head folded into the latent projection: a step reads `[V, rank]`, not the head.
        let (proj, bias) = (
            hidden_out_w.as_dtype(Dtype::Float32)?,
            hidden_out_b.as_dtype(Dtype::Float32)?,
        );
        let (mut rows, mut biases) = (Vec::new(), Vec::new());
        let mut start = 0;
        while start < vocab {
            let end = (start + FOLD_CHUNK).min(vocab);
            let head = target.head_rows_f32(start, end)?;
            let (r, b) = (
                mlx_rs::ops::matmul(&head, &proj)?.as_dtype(Dtype::Bfloat16)?,
                mlx_rs::ops::matmul(&head, &bias)?.as_dtype(Dtype::Bfloat16)?,
            );
            mlx_rs::transforms::eval([&r, &b])?;
            rows.push(r);
            biases.push(b);
            start = end;
        }
        let fold_weight = mlx_rs::ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0)?;
        let fold_bias = mlx_rs::ops::concatenate_axis(&biases.iter().collect::<Vec<_>>(), 0)?;
        let vocab_bias = if args.unpadded_vocab_size < args.vocab_size {
            let mut bias = vec![0.0f32; args.unpadded_vocab_size];
            bias.resize(args.vocab_size, f32::NEG_INFINITY);
            Some(Array::from_slice(&bias, &[vocab]))
        } else {
            None
        };
        let fold_weight_t = mlx_rs::ops::transpose_axes(&fold_weight, &[1, 0])?;
        extra.extend([
            fold_weight.clone(),
            fold_weight_t.clone(),
            fold_bias.clone(),
        ]);
        if let Some(b) = &vocab_bias {
            extra.push(b.clone());
        }
        // Generation runs on another thread than the one that loaded: nothing lazy may outlive the load.
        for batch in held
            .iter()
            .chain(extra.iter())
            .collect::<Vec<_>>()
            .chunks(256)
        {
            mlx_rs::transforms::eval(batch.iter().copied())?;
        }
        Ok(Self {
            args,
            layers,
            norm,
            mask_embed,
            alpha,
            markov: MarkovHead {
                w1,
                rnn_proj,
                rnn_bias,
                context_norm,
                fold_weight_t,
                fold_bias,
                vocab_bias,
            },
        })
    }

    pub fn args(&self) -> &DraftArgs {
        &self.args
    }

    /// An empty context: no position has context rows yet.
    pub fn empty_context(&self) -> DraftContext {
        DraftContext {
            rings: (0..self.layers.len()).map(|_| None).collect(),
            total: 0,
            last_row: None,
        }
    }

    /// A context rebuilt from the taps a conversation kept: the positions `end - len..end`.
    pub fn context_from(&self, taps: &TapRing) -> Result<DraftContext> {
        let mut ctx = self.empty_context();
        ctx.total = taps.end() - taps.len();
        if let Some(rows) = &taps.rows {
            self.extend(&mut ctx, rows)?;
        }
        Ok(ctx)
    }

    /// Adds the taps of the next positions (`[1, n, taps * H]`) to the context: each layer mixes
    /// them with its own weights, projects keys and values, rotates the keys to their absolute
    /// position, and keeps the newest `context_rows`.
    pub fn extend(&self, ctx: &mut DraftContext, rows: &Array) -> Result<()> {
        let a = &self.args;
        let width = (a.taps() * a.hidden_size) as i32;
        ensure!(
            rows.ndim() == 3 && rows.shape()[0] == 1 && rows.shape()[2] == width,
            "taps of shape {:?}, expected [1, n, {width}]",
            rows.shape()
        );
        let n = rows.shape()[1] as usize;
        if n == 0 {
            return Ok(());
        }
        // Only the newest rows are ever attended; the older ones still count as positions.
        let skip = n.saturating_sub(a.context_rows);
        let used = owned_slice(rows, skip as i32, n as i32)?;
        let base = (ctx.total + skip) as i32;
        let (kvh, hd) = (a.num_kv_heads as i32, a.head_dim as i32);
        for (l, layer) in self.layers.iter().enumerate() {
            let mixed = mix_rmsnorm(
                &used,
                &self.alpha[l],
                &layer.context_norm,
                a.taps() as i32,
                a.rms_norm_eps,
            )?;
            let k = split_heads(
                &layer.k_ctx.apply(&mixed)?,
                kvh,
                hd,
                Some((&layer.k_norm_ctx, a.rms_norm_eps)),
            )?;
            let k = self.rope(&k, base)?;
            // Owned, not the transposed view `split_heads` returns: a ring entry outlives the
            // buffer it would otherwise alias.
            let v = super::ops::owned(&split_heads(&layer.v_ctx.apply(&mixed)?, kvh, hd, None)?)?;
            let (k, v) = match ctx.rings[l].take() {
                Some((ok, ov)) => (
                    mlx_rs::ops::concatenate_axis(&[&ok, &k], 2)?,
                    mlx_rs::ops::concatenate_axis(&[&ov, &v], 2)?,
                ),
                None => (k, v),
            };
            let held = k.shape()[2] as usize;
            ctx.rings[l] = Some(if held > a.context_rows {
                let from = (held - a.context_rows) as i32;
                (
                    super::ops::owned(&k.try_index((.., .., from.., ..))?)?,
                    super::ops::owned(&v.try_index((.., .., from.., ..))?)?,
                )
            } else {
                (k, v)
            });
        }
        ctx.total += n;
        ctx.last_row = Some(owned_slice(rows, n as i32 - 1, n as i32)?);
        Ok(())
    }

    fn rope(&self, x: &Array, offset: i32) -> Result<Array> {
        Ok(mlx_rs::fast::rope(
            x,
            self.args.head_dim as i32,
            false,
            Some(self.args.rope_theta),
            1.0,
            offset,
            None,
        )?)
    }

    /// Adds `rows` to the context (when there are any) and proposes the block that follows
    /// `anchor`, the newest committed token, which the target has not fed yet.
    pub fn draft(
        &self,
        ctx: &mut DraftContext,
        rows: Option<&Array>,
        anchor: u32,
        target: &Inkling2Model,
    ) -> Result<DraftOut> {
        if let Some(rows) = rows {
            self.extend(ctx, rows)?;
        }
        self.propose(ctx, anchor, target)
    }

    /// The block after `anchor` (see [`Self::draft`]): `block_size - 1` proposed tokens.
    pub fn propose(
        &self,
        ctx: &DraftContext,
        anchor: u32,
        target: &Inkling2Model,
    ) -> Result<DraftOut> {
        let a = &self.args;
        ensure!(ctx.total > 0, "the draft has no context rows yet");
        let last = ctx
            .last_row
            .as_ref()
            .context("the draft context has no newest row")?;
        let (heads, kvh, hd) = (a.num_heads as i32, a.num_kv_heads as i32, a.head_dim as i32);
        let drafts = a.drafts_per_block() as i32;
        let block = a.block_size as i32;
        let total = ctx.total as i32;

        // The block: the anchor's embedding, then the trained mask vector at every other position.
        let anchor_embed = target.embed_tokens(&[anchor])?;
        let dtype = anchor_embed.dtype();
        let fill = mlx_rs::ops::broadcast_to(
            &mlx_rs::ops::reshape(
                self.mask_embed.as_dtype(dtype)?,
                &[1, 1, a.hidden_size as i32],
            )?,
            &[1, drafts, a.hidden_size as i32],
        )?;
        let mut h = mlx_rs::ops::concatenate_axis(&[&anchor_embed, &fill], 1)?;

        let scale = (hd as f32).powf(-0.5);
        for (l, layer) in self.layers.iter().enumerate() {
            let (ring_k, ring_v) = ctx.rings[l]
                .as_ref()
                .context("a layer has no context rows")?;
            let x = mlx_rs::fast::rms_norm(&h, &layer.input_norm, a.rms_norm_eps)?;
            let q = split_heads(
                &layer.q.apply(&x)?,
                heads,
                hd,
                Some((&layer.q_norm, a.rms_norm_eps)),
            )?;
            let q = self.rope(&q, total)?;
            let bk = split_heads(
                &layer.k.apply(&x)?,
                kvh,
                hd,
                Some((&layer.k_norm, a.rms_norm_eps)),
            )?;
            let bk = self.rope(&bk, total)?;
            let bv = split_heads(&layer.v.apply(&x)?, kvh, hd, None)?;
            let keys = mlx_rs::ops::concatenate_axis(&[ring_k, &bk], 2)?;
            let values = mlx_rs::ops::concatenate_axis(&[ring_v, &bv], 2)?;
            let attended =
                mlx_rs::fast::scaled_dot_product_attention(q, &keys, &values, scale, None, None)?;
            let merged = mlx_rs::ops::reshape(
                mlx_rs::ops::transpose_axes(&attended, &[0, 2, 1, 3])?,
                &[1, block, heads * hd],
            )?;
            h = mlx_rs::ops::add(&h, layer.o.apply(&merged)?)?;
            let x = mlx_rs::fast::rms_norm(&h, &layer.post_attention_norm, a.rms_norm_eps)?;
            let gated = mlx_rs::ops::multiply(silu(&layer.gate.apply(&x)?)?, layer.up.apply(&x)?)?;
            h = mlx_rs::ops::add(&h, layer.down.apply(&gated)?)?;
        }
        let hidden =
            mlx_rs::fast::rms_norm(&h, &self.norm, a.rms_norm_eps)?.try_index((.., 1.., ..))?;
        let base_logits = target.head_logits_raw(&hidden)?;

        // The Markov head, left to right: each position adds a transition from the token before.
        let m = &self.markov;
        let rank = a.markov_rank as i32;
        let rows_f = mlx_rs::ops::reshape(
            mix_rmsnorm(
                last,
                &self.alpha[a.num_layers],
                &m.context_norm,
                a.taps() as i32,
                a.rms_norm_eps,
            )?,
            &[1, a.hidden_size as i32],
        )?
        .as_dtype(hidden.dtype())?;
        let mut state = mlx_rs::ops::zeros_dtype(&[1, rank], hidden.dtype())?;
        let mut prev = Array::from_slice(&[anchor as i32], &[1]);
        let (mut tokens, mut steps) = (Vec::new(), Vec::new());
        for s in 0..drafts {
            let inputs = mlx_rs::ops::concatenate_axis(
                &[
                    &state,
                    &take_axis(&m.w1, &prev, 0)?.as_dtype(hidden.dtype())?,
                    &hidden.try_index((.., s, ..))?,
                    &rows_f,
                ],
                -1,
            )?;
            let proj = mlx_rs::ops::addmm(&m.rnn_bias, &inputs, &m.rnn_proj.0, None, None)?
                .as_dtype(Dtype::Float32)?;
            let part =
                |from: i32, to: i32| -> Result<Array> { Ok(proj.try_index((.., from..to))?) };
            let gate = mlx_rs::ops::sigmoid(part(0, rank)?)?.as_dtype(hidden.dtype())?;
            let kept = mlx_rs::ops::multiply(&gate, &state)?.as_dtype(hidden.dtype())?;
            let blend = mlx_rs::ops::multiply(
                mlx_rs::ops::subtract(Array::from_f32(1.0).as_dtype(hidden.dtype())?, &gate)?,
                mlx_rs::ops::tanh(part(rank, 2 * rank)?)?.as_dtype(hidden.dtype())?,
            )?
            .as_dtype(hidden.dtype())?;
            state = mlx_rs::ops::add(&kept, &blend)?;
            let latent = mlx_rs::ops::tanh(part(2 * rank, 3 * rank)?)?.as_dtype(hidden.dtype())?;
            let bias = mlx_rs::ops::add(
                mlx_rs::ops::matmul(&latent, &m.fold_weight_t)?,
                &m.fold_bias,
            )?;
            let mut logits = mlx_rs::ops::add(
                base_logits
                    .try_index((.., s, ..))?
                    .as_dtype(Dtype::Float32)?,
                bias.as_dtype(Dtype::Float32)?,
            )?;
            if a.final_logit_softcapping > 0.0 {
                let cap = Array::from_f32(a.final_logit_softcapping);
                logits = mlx_rs::ops::multiply(
                    mlx_rs::ops::tanh(mlx_rs::ops::divide(&logits, &cap)?)?,
                    &cap,
                )?;
            }
            if let Some(vb) = &m.vocab_bias {
                logits = mlx_rs::ops::add(&logits, vb)?;
            }
            prev = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false)?;
            tokens.push(prev.clone());
            steps.push(logits);
        }
        let ids = mlx_rs::ops::concatenate_axis(&tokens.iter().collect::<Vec<_>>(), 0)?;
        ids.eval()?;
        let tokens: Vec<u32> = ids.as_dtype(Dtype::Uint32)?.as_slice::<u32>().to_vec();
        if tokens.len() != drafts as usize {
            bail!(
                "the draft proposed {} tokens, expected {drafts}",
                tokens.len()
            );
        }
        let step_logits = mlx_rs::ops::expand_dims(
            mlx_rs::ops::concatenate_axis(&steps.iter().collect::<Vec<_>>(), 0)?,
            0,
        )?;
        Ok(DraftOut {
            tokens,
            hidden,
            base_logits,
            step_logits,
        })
    }
}

fn silu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::multiply(x, mlx_rs::ops::sigmoid(x)?)?)
}
