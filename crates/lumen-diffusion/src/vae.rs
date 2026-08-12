//! Native FLUX.2 VAE decoder (latent → image), Phase 0 of the diffusion port.
//!
//! A faithful mechanical port of the diffusers / mflux `AutoencoderKL` decoder
//! for FLUX.2-klein-4B. The decoder stays **channels-last** `[B, H, W, C]`
//! throughout (matching the native [`crate::ops::conv2d`] / GroupNorm layout);
//! the channels-first latent is transposed once at input and the result is left
//! channels-last for the caller to transpose back if it wants NCHW.
//!
//! Architecture (config: latent_channels=32, block_out_channels=(128,256,512,
//! 512), layers_per_block=2 → 3 resnets per up-block, norm_num_groups=32,
//! eps=1e-6, out_channels=3, scaling_factor=1.0, shift_factor=0.0):
//!
//! ```text
//! decode(latent[B,32,H,W]):
//!   post_quant_conv (1x1, 32->32)
//!   conv_in (3x3 pad1, 32->512)
//!   mid_block: resnet -> attention -> resnet           (512)
//!   up_blocks (reversed channels [512,512,256,128]):
//!     block0 512->512, +upsample
//!     block1 512->512, +upsample
//!     block2 512->256, +upsample   (resnets.0 has conv_shortcut)
//!     block3 256->128, no upsample (resnets.0 has conv_shortcut)
//!   group_norm(32) -> silu -> conv_out (3x3 pad1, 128->3)
//! ```
//!
//! Weights are loaded from the diffusers BF16 `safetensors`. PyTorch conv layout
//! `[Cout,Cin,kH,kW]` is transposed to mlx `[Cout,kH,kW,Cin]` (axes 0,2,3,1);
//! `Linear` weights `[out,in]` are transposed to `[in,out]` for `matmul`.
//!
//! Design docs: `.ai/memory/active/flux2-dev-diffusion-port/`.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::collections::HashMap;
    use std::path::Path;

    use anyhow::{Context, Result, anyhow, bail};
    use mlx_rs::Array;

    use crate::ops::conv2d::conv2d;
    use crate::ops::groupnorm::group_norm;
    use crate::ops::linear::linear;
    use crate::ops::silu::silu;

    const NUM_GROUPS: i32 = 32;
    const EPS: f32 = 1e-6;
    /// BatchNorm eps for the packed-latent denorm (mflux `Flux2BatchNormStats`).
    const BN_EPS: f32 = 1e-4;

    // ── safetensors loader ────────────────────────────────────────────────────

    struct TensorMeta {
        dtype: String,
        shape: Vec<i32>,
        start: usize,
        end: usize,
    }

    /// A loaded safetensors blob: JSON header metadata + raw tensor bytes.
    pub struct SafeTensors {
        metas: HashMap<String, TensorMeta>,
        data_start: usize,
        blob: Vec<u8>,
    }

    impl SafeTensors {
        /// Parse `8-byte LE header length + JSON header + tensor bytes`.
        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            let blob = std::fs::read(path.as_ref())
                .with_context(|| format!("reading safetensors {:?}", path.as_ref()))?;
            if blob.len() < 8 {
                bail!("safetensors file too short");
            }
            let hlen = u64::from_le_bytes(blob[0..8].try_into().unwrap()) as usize;
            let header_end = 8 + hlen;
            if blob.len() < header_end {
                bail!("safetensors header length {hlen} exceeds file size");
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

        pub fn has(&self, name: &str) -> bool {
            self.metas.contains_key(name)
        }

        /// Load a tensor as an f32 [`Array`] in its native (PyTorch) shape.
        fn raw(&self, name: &str) -> Result<Array> {
            let m = self
                .metas
                .get(name)
                .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
            let bytes = &self.blob[self.data_start + m.start..self.data_start + m.end];
            let data: Vec<f32> = match m.dtype.as_str() {
                "BF16" => {
                    if bytes.len() % 2 != 0 {
                        bail!("BF16 tensor {name} has odd byte length");
                    }
                    bytes
                        .chunks_exact(2)
                        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                        .collect()
                }
                "F32" => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                "F16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                other => bail!("unsupported dtype {other} for tensor {name}"),
            };
            Ok(Array::from_slice(&data, &m.shape))
        }

        /// Conv weight: PyTorch `[Cout,Cin,kH,kW]` → mlx `[Cout,kH,kW,Cin]`.
        fn conv_weight(&self, name: &str) -> Result<Array> {
            let w = self.raw(name)?;
            let w = w
                .transpose_axes(&[0, 2, 3, 1])
                .with_context(|| format!("conv weight transpose {name}"))?;
            w.eval()
                .with_context(|| format!("eval conv weight {name}"))?;
            Ok(w)
        }

        /// Linear weight: diffusers `[out,in]` → mlx `[in,out]` for `matmul`.
        fn linear_weight(&self, name: &str) -> Result<Array> {
            let w = self.raw(name)?;
            let w = w
                .transpose_axes(&[1, 0])
                .with_context(|| format!("linear weight transpose {name}"))?;
            w.eval()
                .with_context(|| format!("eval linear weight {name}"))?;
            Ok(w)
        }

        /// 1-D vector (bias / norm weight) loaded as-is.
        fn vec(&self, name: &str) -> Result<Array> {
            let v = self.raw(name)?;
            v.eval().with_context(|| format!("eval vec {name}"))?;
            Ok(v)
        }
    }

    // ── weight bundles ────────────────────────────────────────────────────────

    struct ConvW {
        weight: Array,
        bias: Array,
    }

    impl ConvW {
        fn load(st: &SafeTensors, prefix: &str) -> Result<Self> {
            Ok(Self {
                weight: st.conv_weight(&format!("{prefix}.weight"))?,
                bias: st.vec(&format!("{prefix}.bias"))?,
            })
        }
    }

    struct NormW {
        weight: Array,
        bias: Array,
    }

    impl NormW {
        fn load(st: &SafeTensors, prefix: &str) -> Result<Self> {
            Ok(Self {
                weight: st.vec(&format!("{prefix}.weight"))?,
                bias: st.vec(&format!("{prefix}.bias"))?,
            })
        }
    }

    struct ResnetW {
        norm1: NormW,
        conv1: ConvW,
        norm2: NormW,
        conv2: ConvW,
        shortcut: Option<ConvW>,
        c_in: i32,
        c_out: i32,
    }

    impl ResnetW {
        fn load(st: &SafeTensors, prefix: &str, c_in: i32, c_out: i32) -> Result<Self> {
            let shortcut = if c_in != c_out {
                Some(ConvW::load(st, &format!("{prefix}.conv_shortcut"))?)
            } else {
                None
            };
            Ok(Self {
                norm1: NormW::load(st, &format!("{prefix}.norm1"))?,
                conv1: ConvW::load(st, &format!("{prefix}.conv1"))?,
                norm2: NormW::load(st, &format!("{prefix}.norm2"))?,
                conv2: ConvW::load(st, &format!("{prefix}.conv2"))?,
                shortcut,
                c_in,
                c_out,
            })
        }
    }

    struct LinearW {
        weight: Array, // [in, out]
        bias: Array,
    }

    impl LinearW {
        fn load(st: &SafeTensors, prefix: &str) -> Result<Self> {
            Ok(Self {
                weight: st.linear_weight(&format!("{prefix}.weight"))?,
                bias: st.vec(&format!("{prefix}.bias"))?,
            })
        }
    }

    struct AttnW {
        group_norm: NormW,
        to_q: LinearW,
        to_k: LinearW,
        to_v: LinearW,
        to_out: LinearW,
    }

    impl AttnW {
        fn load(st: &SafeTensors, prefix: &str, _channels: i32) -> Result<Self> {
            Ok(Self {
                group_norm: NormW::load(st, &format!("{prefix}.group_norm"))?,
                to_q: LinearW::load(st, &format!("{prefix}.to_q"))?,
                to_k: LinearW::load(st, &format!("{prefix}.to_k"))?,
                to_v: LinearW::load(st, &format!("{prefix}.to_v"))?,
                to_out: LinearW::load(st, &format!("{prefix}.to_out.0"))?,
            })
        }
    }

    struct UpBlockW {
        resnets: Vec<ResnetW>,
        upsampler: Option<ConvW>,
    }

    /// All decoder weights for FLUX.2-klein.
    pub struct VaeDecoder {
        post_quant_conv: ConvW,
        conv_in: ConvW,
        mid_resnet0: ResnetW,
        mid_attn: AttnW,
        mid_resnet1: ResnetW,
        up_blocks: Vec<UpBlockW>,
        conv_norm_out: NormW,
        conv_out: ConvW,
        /// Packed-latent BatchNorm denorm stats (`bn.running_mean/var`, [128]).
        bn_mean: Array,
        bn_std: Array, // sqrt(running_var + BN_EPS)
    }

    impl VaeDecoder {
        /// Load all decoder + `post_quant_conv` weights from a diffusers
        /// `safetensors` file.
        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            let st = SafeTensors::load(path)?;

            // reversed block_out_channels = [512, 512, 256, 128]
            let block_io: [(i32, i32); 4] = [(512, 512), (512, 512), (512, 256), (256, 128)];
            let has_upsampler = [true, true, true, false];

            let mut up_blocks = Vec::with_capacity(4);
            for (bi, ((c_in, c_out), &up)) in block_io.iter().zip(has_upsampler.iter()).enumerate()
            {
                let p = format!("decoder.up_blocks.{bi}");
                let mut resnets = Vec::with_capacity(3);
                for ri in 0..3 {
                    // resnet.0 maps c_in->c_out; resnets.1/2 map c_out->c_out.
                    let ci = if ri == 0 { *c_in } else { *c_out };
                    resnets.push(ResnetW::load(
                        &st,
                        &format!("{p}.resnets.{ri}"),
                        ci,
                        *c_out,
                    )?);
                }
                let upsampler = if up {
                    Some(ConvW::load(&st, &format!("{p}.upsamplers.0.conv"))?)
                } else {
                    None
                };
                up_blocks.push(UpBlockW { resnets, upsampler });
            }

            // Packed-latent BN denorm stats: [128] → [1,128,1,1] for the
            // channels-first `packed * std + mean`. eps folded into std here.
            let bn_mean = st
                .vec("bn.running_mean")?
                .reshape(&[1, 128, 1, 1])
                .context("bn_mean reshape")?;
            let bn_var = st.vec("bn.running_var")?;
            let bn_eps = Array::from_slice(&[BN_EPS], &[1]);
            let bn_std = bn_var
                .add(&bn_eps)
                .and_then(|v| v.sqrt())
                .and_then(|v| v.reshape(&[1, 128, 1, 1]))
                .context("bn_std = sqrt(var+eps) reshape")?;
            bn_mean.eval().context("eval bn_mean")?;
            bn_std.eval().context("eval bn_std")?;

            Ok(Self {
                post_quant_conv: ConvW::load(&st, "post_quant_conv")?,
                conv_in: ConvW::load(&st, "decoder.conv_in")?,
                mid_resnet0: ResnetW::load(&st, "decoder.mid_block.resnets.0", 512, 512)?,
                mid_attn: AttnW::load(&st, "decoder.mid_block.attentions.0", 512)?,
                mid_resnet1: ResnetW::load(&st, "decoder.mid_block.resnets.1", 512, 512)?,
                up_blocks,
                conv_norm_out: NormW::load(&st, "decoder.conv_norm_out")?,
                conv_out: ConvW::load(&st, "decoder.conv_out")?,
                bn_mean,
                bn_std,
            })
        }

        /// Decode a channels-last latent `[B, H, W, 32]` → image `[B, Ho, Wo, 3]`.
        ///
        /// Caller is responsible for transposing a channels-first
        /// `[B, 32, H, W]` latent into channels-last before calling.
        pub fn decode(&self, latent_chlast: &Array) -> Result<Array> {
            // post_quant_conv (1x1, no pad) then conv_in (3x3 pad1).
            let mut h = conv_bias(latent_chlast, &self.post_quant_conv, (0, 0))?;
            h = conv_bias(&h, &self.conv_in, (1, 1))?;

            // mid block
            h = resnet_forward(&h, &self.mid_resnet0)?;
            h = attn_forward(&h, &self.mid_attn)?;
            h = resnet_forward(&h, &self.mid_resnet1)?;

            // up blocks
            for block in &self.up_blocks {
                for rb in &block.resnets {
                    h = resnet_forward(&h, rb)?;
                }
                if let Some(up) = &block.upsampler {
                    h = upsample_forward(&h, up)?;
                }
            }

            // out
            h = group_norm(
                &h,
                NUM_GROUPS,
                Some(&self.conv_norm_out.weight),
                Some(&self.conv_norm_out.bias),
                EPS,
                true,
            )?;
            h = silu(&h)?;
            h = conv_bias(&h, &self.conv_out, (1, 1))?;
            h.eval().context("vae decode: final eval failed")?;
            Ok(h)
        }

        /// Decode packed FLUX.2 latents → image `[B, Ho, Wo, 3]` (channels-last).
        ///
        /// Mirrors mflux `Flux2VAE.decode_packed_latents`:
        /// 1. `packed_cf` arrives channels-first `[B, 128, lh, lw]` (the loop's
        ///    final `reshape(B,lh,lw,128).transpose(0,3,1,2)`).
        /// 2. BN denorm: `latents = packed * sqrt(var+eps) + mean`.
        /// 3. `_unpatchify_latents`: `[B,128,h,w] → reshape [B,32,2,2,h,w] →
        ///    transpose (0,1,4,2,5,3) → reshape [B,32,2h,2w]`.
        /// 4. channels-first → channels-last, then [`decode`] (post_quant_conv +
        ///    decoder).
        pub fn decode_packed_latents(&self, packed_cf: &Array) -> Result<Array> {
            // shape sanity: [B,128,lh,lw]
            let sh = packed_cf.shape().to_vec();
            if sh.len() != 4 || sh[1] != 128 {
                bail!("decode_packed_latents expects [B,128,lh,lw], got {sh:?}");
            }
            let (b, h, w) = (sh[0], sh[2], sh[3]);

            // 2. BN denorm (channels-first broadcast over [1,128,1,1]).
            let denorm = packed_cf
                .multiply(&self.bn_std)
                .and_then(|x| x.add(&self.bn_mean))
                .context("bn denorm failed")?;

            // 3. unpatchify: [B,128,h,w] → [B,32,2,2,h,w] → (0,1,4,2,5,3) →
            //    [B,32,2h,2w].
            let up = denorm
                .reshape(&[b, 32, 2, 2, h, w])
                .and_then(|x| x.transpose_axes(&[0, 1, 4, 2, 5, 3]))
                .and_then(|x| x.reshape(&[b, 32, h * 2, w * 2]))
                .context("unpatchify failed")?;

            // 4. channels-first → channels-last for the existing decoder.
            let latent_cl = up
                .transpose_axes(&[0, 2, 3, 1])
                .context("packed latent → channels-last failed")?;
            latent_cl.eval().context("eval packed latent_cl")?;
            self.decode(&latent_cl)
        }

        /// Debug: bn-denorm + unpatchify only, returned channels-first
        /// `[B,32,2lh,2lw]` — isolates the packed-latent glue from the deep
        /// decoder (whose mflux reference runs in bf16 and is non-deterministic).
        #[cfg(test)]
        pub fn unpatchify_debug(&self, packed_cf: &Array) -> Result<Array> {
            let sh = packed_cf.shape().to_vec();
            let (b, h, w) = (sh[0], sh[2], sh[3]);
            let denorm = packed_cf
                .multiply(&self.bn_std)
                .and_then(|x| x.add(&self.bn_mean))?;
            let up = denorm
                .reshape(&[b, 32, 2, 2, h, w])
                .and_then(|x| x.transpose_axes(&[0, 1, 4, 2, 5, 3]))
                .and_then(|x| x.reshape(&[b, 32, h * 2, w * 2]))?;
            up.eval()?;
            Ok(up)
        }

        /// Debug: return (after_conv_in, after_mid_resnet0, after_mid_attn),
        /// all channels-last. Used only by parity debugging.
        #[cfg(test)]
        pub fn decode_debug_stages(&self, latent_chlast: &Array) -> Result<(Array, Array, Array)> {
            let mut h = conv_bias(latent_chlast, &self.post_quant_conv, (0, 0))?;
            h = conv_bias(&h, &self.conv_in, (1, 1))?;
            let after_convin = h.clone();
            h = resnet_forward(&h, &self.mid_resnet0)?;
            let after_res0 = h.clone();
            h = attn_forward(&h, &self.mid_attn)?;
            let after_attn = h.clone();
            after_convin.eval()?;
            after_res0.eval()?;
            after_attn.eval()?;
            Ok((after_convin, after_res0, after_attn))
        }
    }

    // ── forward helpers ───────────────────────────────────────────────────────

    /// conv2d (cross-correlation, stride 1, given padding) + channel-last bias.
    fn conv_bias(x: &Array, w: &ConvW, padding: (i32, i32)) -> Result<Array> {
        let y = conv2d(x, &w.weight, (1, 1), padding, (1, 1), 1)?;
        y.add(&w.bias).context("conv_bias: + bias failed")
    }

    fn resnet_forward(x: &Array, w: &ResnetW) -> Result<Array> {
        let mut h = group_norm(
            x,
            NUM_GROUPS,
            Some(&w.norm1.weight),
            Some(&w.norm1.bias),
            EPS,
            true,
        )?;
        h = silu(&h)?;
        h = conv_bias(&h, &w.conv1, (1, 1))?;
        h = group_norm(
            &h,
            NUM_GROUPS,
            Some(&w.norm2.weight),
            Some(&w.norm2.bias),
            EPS,
            true,
        )?;
        h = silu(&h)?;
        h = conv_bias(&h, &w.conv2, (1, 1))?;

        let residual = if let Some(sc) = &w.shortcut {
            // 1x1 conv, no padding.
            conv_bias(x, sc, (0, 0))?
        } else {
            debug_assert_eq!(w.c_in, w.c_out);
            x.clone()
        };
        h.add(&residual).context("resnet: h + residual failed")
    }

    /// Single-head self-attention over the spatial grid, channels-last.
    fn attn_forward(x: &Array, w: &AttnW) -> Result<Array> {
        let shape = x.shape().to_vec(); // [B, H, W, C]
        let (b, hh, ww, c) = (shape[0], shape[1], shape[2], shape[3]);
        let hw = hh * ww;

        let normed = group_norm(
            x,
            NUM_GROUPS,
            Some(&w.group_norm.weight),
            Some(&w.group_norm.bias),
            EPS,
            true,
        )?;
        // [B, H*W, C]
        let flat = normed
            .reshape(&[b, hw, c])
            .context("attn: flatten spatial failed")?;

        let q = linear(&flat, &w.to_q.weight, &w.to_q.bias)?;
        let k = linear(&flat, &w.to_k.weight, &w.to_k.bias)?;
        let v = linear(&flat, &w.to_v.weight, &w.to_v.bias)?;

        // [B, H*W, C] -> [B, 1, H*W, C]  (single head, head_dim = C)
        let to_bhqc = |t: &Array| -> Result<Array> {
            t.reshape(&[b, hw, 1, c])
                .and_then(|r| r.transpose_axes(&[0, 2, 1, 3]))
                .context("attn: reshape to [B,1,HW,C] failed")
        };
        let q = to_bhqc(&q)?;
        let k = to_bhqc(&k)?;
        let v = to_bhqc(&v)?;

        let scale = 1.0f32 / (c as f32).sqrt();
        let attended = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)
            .context("attn: scaled_dot_product_attention failed")?;

        // [B,1,HW,C] -> [B,HW,1,C] -> [B,HW,C]
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|t| t.reshape(&[b, hw, c]))
            .context("attn: reshape back failed")?;

        let out = linear(&attended, &w.to_out.weight, &w.to_out.bias)?;
        let out = out
            .reshape(&[b, hh, ww, c])
            .context("attn: reshape to [B,H,W,C] failed")?;
        x.add(&out).context("attn: residual add failed")
    }

    /// Nearest 2× upsample (channels-last: repeat H axis 1, W axis 2) then conv.
    fn upsample_forward(x: &Array, conv: &ConvW) -> Result<Array> {
        // repeat takes Array by value; clone the (cheap, lazy) handle.
        let up_h = mlx_rs::ops::repeat_axis::<f32>(x.clone(), 2, 1)
            .context("upsample: repeat H failed")?;
        let up =
            mlx_rs::ops::repeat_axis::<f32>(up_h, 2, 2).context("upsample: repeat W failed")?;
        conv_bias(&up, conv, (1, 1))
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{SafeTensors, VaeDecoder};

// ── parity test against the numpy golden reference (/tmp/vae_*.bin) ───────────
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::VaeDecoder;
    use mlx_rs::Array;

    // Klein VAE, resolved from the local HF cache (shared dev/klein FLUX.2 VAE).
    fn vae_path() -> std::path::PathBuf {
        crate::hf_cache::snapshot_path_or_rel(
            crate::repos::VAE,
            "vae/diffusion_pytorch_model.safetensors",
        )
    }

    /// Reorder a channels-first `[B,C,H,W]` flat buffer into channels-last
    /// `[B,H,W,C]` flat order — pure index math, so we never have to materialize
    /// a transposed mlx array (mlx `as_slice` reads the physical buffer and
    /// ignores a lazy transpose's strides). The native decoder output is already
    /// contiguous channels-last, so comparing in channels-last space is exact.
    fn cf_to_cl(cf: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cf.len()];
        for bi in 0..b {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let src = ((bi * c + ci) * h + hi) * w + wi;
                        let dst = ((bi * h + hi) * w + wi) * c + ci;
                        out[dst] = cf[src];
                    }
                }
            }
        }
        out
    }

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Compare a native channels-last `[B,H,W,C]` array against a channels-first
    /// reference bin, in channels-last space (no mlx transpose involved).
    fn cmp_chlast_vs_cf(
        label: &str,
        cl: &Array,
        cf_path: &str,
        b: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> f32 {
        cl.eval().expect("eval cl");
        let obs: &[f32] = cl.as_slice(); // contiguous channels-last
        let exp_cf = read_f32_bin(cf_path);
        let exp = cf_to_cl(&exp_cf, b as usize, c as usize, h as usize, w as usize);
        assert_eq!(obs.len(), (b * c * h * w) as usize, "{label} size");
        assert_eq!(obs.len(), exp.len(), "{label} size vs ref");
        let mut max_abs = 0.0f32;
        let mut sum = 0.0f64;
        for (&g, &e) in obs.iter().zip(exp.iter()) {
            let d = (g - e).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum += d as f64;
        }
        println!(
            "{label}: max_abs={max_abs:.6e} mean_abs={:.6e}",
            sum / obs.len() as f64
        );
        max_abs
    }

    #[test]
    #[ignore = "needs a Metal device AND /tmp/vae_latent.bin plus the \
                /tmp/dbg_after_*.bin stage dumps; no committed script produces \
                them — see crates/lumen-mlx/tests/golden/README.md"]
    fn vae_decoder_debug_stages() {
        let latent_cf = read_f32_bin("/tmp/vae_latent.bin");
        let latent = Array::from_slice(&latent_cf, &[1, 32, 16, 16]);
        let latent_cl = latent.transpose_axes(&[0, 2, 3, 1]).unwrap();
        latent_cl.eval().unwrap();
        let decoder = VaeDecoder::load(vae_path()).expect("load VAE weights");
        let (a, b, c) = decoder.decode_debug_stages(&latent_cl).unwrap();
        cmp_chlast_vs_cf(
            "after_conv_in",
            &a,
            "/tmp/dbg_after_convin.bin",
            1,
            512,
            16,
            16,
        );
        cmp_chlast_vs_cf(
            "after_mid_res0",
            &b,
            "/tmp/dbg_after_midres0.bin",
            1,
            512,
            16,
            16,
        );
        cmp_chlast_vs_cf(
            "after_mid_attn",
            &c,
            "/tmp/dbg_after_midattn.bin",
            1,
            512,
            16,
            16,
        );
    }

    /// Golden parity: numpy reference decode (channels-first) vs native mlx-rs
    /// decode (channels-last). Run `/tmp/vae_ref.py` first to produce the bins.
    #[test]
    #[ignore = "needs a Metal device AND /tmp/vae_{latent,out}.bin reference \
                dumps; no committed script produces them — see \
                crates/lumen-mlx/tests/golden/README.md"]
    fn vae_decoder_matches_reference() {
        // latent [1,32,16,16] channels-first.
        let latent_cf = read_f32_bin("/tmp/vae_latent.bin");
        assert_eq!(latent_cf.len(), 1 * 32 * 16 * 16, "latent size");
        let expected = read_f32_bin("/tmp/vae_out.bin"); // [1,3,128,128] channels-first
        assert_eq!(expected.len(), 1 * 3 * 128 * 128, "expected size");

        // Build channels-first array then transpose to channels-last [1,16,16,32].
        let latent = Array::from_slice(&latent_cf, &[1, 32, 16, 16]);
        let latent_cl = latent
            .transpose_axes(&[0, 2, 3, 1])
            .expect("latent transpose to channels-last");
        latent_cl.eval().expect("eval latent");

        let decoder = VaeDecoder::load(vae_path()).expect("load VAE weights");
        let out_cl = decoder.decode(&latent_cl).expect("decode"); // [1,128,128,3]
        assert_eq!(
            out_cl.shape(),
            &[1, 128, 128, 3],
            "output shape (channels-last)"
        );
        out_cl.eval().expect("eval output");
        let observed: &[f32] = out_cl.as_slice(); // contiguous channels-last

        // Reference is channels-first [1,3,128,128]; reorder to channels-last
        // for an apples-to-apples compare (avoids materializing a transpose).
        let expected_cl = cf_to_cl(&expected, 1, 3, 128, 128);
        assert_eq!(observed.len(), expected_cl.len());

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (&got, &exp) in observed.iter().zip(expected_cl.iter()) {
            let d = (got - exp).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / observed.len() as f64) as f32;
        println!("VAE parity: max_abs_diff={max_abs:.6e}  mean_abs_diff={mean_abs:.6e}");

        assert!(
            max_abs < 5e-2,
            "VAE decode max-abs diff {max_abs:.6e} exceeds 5e-2 tolerance (mean {mean_abs:.6e})"
        );
    }
}
