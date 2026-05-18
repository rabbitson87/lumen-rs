//! Stateless Qwen3 transformer — single-pass forward, no KV cache.
//!
//! Mirrors the structure of `candle_transformers::models::qwen3` but
//! removes the `ConcatKvCache` accumulator and exposes `&self` forward
//! (no `&mut`). For embedding inference each input is independent —
//! cache reuse is wasted work and the `mut` requirement forces external
//! locking.
//!
//! Supports two projection backends via [`LinearKind`]:
//!   - `Plain` — `candle_nn::Linear` over a fully materialized bf16 weight.
//!   - `Quant` — [`lumen_metal::affine8_linear::Affine8Linear`] over an
//!     MLX 8-bit packed weight (zero CPU dequant; matmul + dequant fused
//!     in a Metal kernel).
//!
//! Layer naming matches HF Qwen3ForCausalLM minus the lm_head:
//!   model.embed_tokens.weight
//!   model.layers.{i}.input_layernorm.weight
//!   model.layers.{i}.post_attention_layernorm.weight
//!   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   model.layers.{i}.self_attn.{q,k}_norm.weight
//!   model.layers.{i}.mlp.{gate,up,down}_proj.weight
//!   model.norm.weight

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear_b, linear_no_bias};
use candle_transformers::models::qwen3::Config;
use candle_transformers::utils::repeat_kv;
use lumen_metal::affine8_linear::Affine8Linear;

#[derive(Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

/// A linear projection — either a plain candle `Linear` over dequanted
/// bf16 weights, or an `Affine8Linear` that runs an MLX 8-bit packed
/// weight through a fused Metal dequant+matmul kernel.
pub enum LinearKind {
    Plain(Linear),
    Quant(Affine8Linear),
}

impl LinearKind {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Plain(l) => candle_core::Module::forward(l, x),
            Self::Quant(q) => q
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
        }
    }
}

struct Mlp {
    gate: LinearKind,
    up: LinearKind,
    down: LinearKind,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.silu()?;
        let u = self.up.forward(x)?;
        let h = (g * u)?;
        Ok(self.down.forward(&h)?)
    }
}

struct Attn {
    q_proj: LinearKind,
    k_proj: LinearKind,
    v_proj: LinearKind,
    o_proj: LinearKind,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    rms_eps: f32,
    n_heads: usize,
    n_kv: usize,
    n_kv_groups: usize,
    head_dim: usize,
    hidden: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attn {
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.n_kv, self.head_dim))?.transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = candle_nn::ops::rms_norm(&q_flat, &self.q_norm_w, self.rms_eps)?;
        let k_flat = candle_nn::ops::rms_norm(&k_flat, &self.k_norm_w, self.rms_eps)?;
        let q = q_flat.reshape((b, self.n_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.n_kv, l, self.head_dim))?;

        let (q, k) = self.rotary.apply(&q, &k, 0)?;

        let k = repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden))?;
        Ok(self.o_proj.forward(&ctx)?)
    }
}

struct Layer {
    attn: Attn,
    mlp: Mlp,
    ln1_w: Tensor,
    ln2_w: Tensor,
    rms_eps: f32,
}

impl Layer {
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = candle_nn::ops::rms_norm(x, &self.ln1_w, self.rms_eps)?;
        let h = self.attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h = candle_nn::ops::rms_norm(&x, &self.ln2_w, self.rms_eps)?;
        let h = self.mlp.forward(&h)?;
        Ok((x + h)?)
    }
}

pub struct StatelessQwen3 {
    embed: Embedding,
    layers: Vec<Layer>,
    norm_w: Tensor,
    rms_eps: f32,
    device: Device,
    dtype: DType,
}

impl StatelessQwen3 {
    /// Build with all projections materialized as plain bf16 weights.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vbl = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            let vbi = vbl.pp(i);
            let attn_vb = vbi.pp("self_attn");
            let mlp_vb = vbi.pp("mlp");
            let head_dim = cfg.head_dim;
            let attn = Attn {
                q_proj: LinearKind::Plain(linear_b(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    cfg.attention_bias,
                    attn_vb.pp("q_proj"),
                )?),
                k_proj: LinearKind::Plain(linear_b(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    cfg.attention_bias,
                    attn_vb.pp("k_proj"),
                )?),
                v_proj: LinearKind::Plain(linear_b(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    cfg.attention_bias,
                    attn_vb.pp("v_proj"),
                )?),
                o_proj: LinearKind::Plain(linear_b(
                    cfg.num_attention_heads * head_dim,
                    cfg.hidden_size,
                    cfg.attention_bias,
                    attn_vb.pp("o_proj"),
                )?),
                q_norm_w: attn_vb.pp("q_norm").get(head_dim, "weight")?,
                k_norm_w: attn_vb.pp("k_norm").get(head_dim, "weight")?,
                rms_eps: cfg.rms_norm_eps as f32,
                n_heads: cfg.num_attention_heads,
                n_kv: cfg.num_key_value_heads,
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                head_dim,
                hidden: head_dim * cfg.num_attention_heads,
                rotary: rotary.clone(),
            };
            let mlp = Mlp {
                gate: LinearKind::Plain(linear_no_bias(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    mlp_vb.pp("gate_proj"),
                )?),
                up: LinearKind::Plain(linear_no_bias(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    mlp_vb.pp("up_proj"),
                )?),
                down: LinearKind::Plain(linear_no_bias(
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    mlp_vb.pp("down_proj"),
                )?),
            };
            layers.push(Layer {
                attn,
                mlp,
                ln1_w: vbi.pp("input_layernorm").get(cfg.hidden_size, "weight")?,
                ln2_w: vbi.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?,
                rms_eps: cfg.rms_norm_eps as f32,
            });
        }
        Ok(Self {
            embed,
            layers,
            norm_w: vb.pp("model.norm").get(cfg.hidden_size, "weight")?,
            rms_eps: cfg.rms_norm_eps as f32,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Build with quantized projections supplied via a HashMap keyed by
    /// HF param path (e.g. `"model.layers.0.self_attn.q_proj"`). The
    /// embed/norm weights still come from `vb` (which the caller has
    /// already populated with the dequantized embed_tokens + bf16 norm
    /// weights).
    pub fn new_quantized(
        cfg: &Config,
        vb: VarBuilder,
        mut projections: HashMap<String, Affine8Linear>,
    ) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vbl = vb.pp("model.layers");

        let take = |projections: &mut HashMap<String, Affine8Linear>, key: &str| -> Result<LinearKind> {
            projections
                .remove(key)
                .map(LinearKind::Quant)
                .ok_or_else(|| anyhow!("missing quant projection: {key}"))
        };

        for i in 0..cfg.num_hidden_layers {
            let vbi = vbl.pp(i);
            let attn_vb = vbi.pp("self_attn");
            let head_dim = cfg.head_dim;
            let attn = Attn {
                q_proj: take(&mut projections, &format!("model.layers.{i}.self_attn.q_proj"))?,
                k_proj: take(&mut projections, &format!("model.layers.{i}.self_attn.k_proj"))?,
                v_proj: take(&mut projections, &format!("model.layers.{i}.self_attn.v_proj"))?,
                o_proj: take(&mut projections, &format!("model.layers.{i}.self_attn.o_proj"))?,
                q_norm_w: attn_vb.pp("q_norm").get(head_dim, "weight")?,
                k_norm_w: attn_vb.pp("k_norm").get(head_dim, "weight")?,
                rms_eps: cfg.rms_norm_eps as f32,
                n_heads: cfg.num_attention_heads,
                n_kv: cfg.num_key_value_heads,
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                head_dim,
                hidden: head_dim * cfg.num_attention_heads,
                rotary: rotary.clone(),
            };
            let mlp = Mlp {
                gate: take(&mut projections, &format!("model.layers.{i}.mlp.gate_proj"))?,
                up: take(&mut projections, &format!("model.layers.{i}.mlp.up_proj"))?,
                down: take(&mut projections, &format!("model.layers.{i}.mlp.down_proj"))?,
            };
            layers.push(Layer {
                attn,
                mlp,
                ln1_w: vbi.pp("input_layernorm").get(cfg.hidden_size, "weight")?,
                ln2_w: vbi.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?,
                rms_eps: cfg.rms_norm_eps as f32,
            });
        }
        Ok(Self {
            embed,
            layers,
            norm_w: vb.pp("model.norm").get(cfg.hidden_size, "weight")?,
            rms_eps: cfg.rms_norm_eps as f32,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn causal_mask(&self, l: usize) -> Result<Tensor> {
        if l <= 1 {
            return Ok(Tensor::zeros((1, 1, l, l), self.dtype, &self.device)?);
        }
        let minf = f32::NEG_INFINITY;
        let mut buf: Vec<f32> = Vec::with_capacity(l * l);
        for i in 0..l {
            for j in 0..l {
                buf.push(if j <= i { 0.0 } else { minf });
            }
        }
        Ok(Tensor::from_vec(buf, (1, 1, l, l), &self.device)?.to_dtype(self.dtype)?)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let mut h = candle_core::Module::forward(&self.embed, input)?;
        let mask = if l > 1 { Some(self.causal_mask(l)?) } else { None };
        for layer in &self.layers {
            h = layer.forward(&h, mask.as_ref())?;
        }
        Ok(candle_nn::ops::rms_norm(&h, &self.norm_w, self.rms_eps)?)
    }
}
