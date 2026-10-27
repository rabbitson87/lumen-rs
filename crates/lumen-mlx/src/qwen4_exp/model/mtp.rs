//! The MTP (multi-token-prediction) draft head of Qwen3.8-Flash-Next.
//!
//! One extra full-attention decoder layer (QSA attention, a resident 512-expert MoE and
//! the hyper-connections) that predicts the token AFTER the next one, used for
//! self-speculative decoding. Semantics follow vLLM's `Qwen4ExpMultiTokenPredictor`
//! ("scheme A"), through Slotstream's `Tools/reference/mtp_ref.py`:
//!
//!   fuse:  e = fc_embedding(rmsnorm(embed(token)))                       `[1, S, H]`
//!          h = fc_hidden(rmsnorm_fullwidth(multi)) per branch            `[1, S, hc, H]`
//!          x = flatten(e broadcast over the branches + h)                `[1, S, hc*H]`
//!   layer: one full-attention layer (the model's own `Layer`)
//!   out:   its own hyper-connection mixer(x) -> `[1, S, H]` for the shared lm_head, and
//!          the pre-mixer `x` as the "multi" a chained draft step would take
//!
//! `multi` is the main model's pre-final-mixer residual stream (`hc*H` wide) at the
//! position before the token being drafted from. The head keeps its own KV cache and
//! indexer keys, one entry per consumed token except the first: entry `i` fuses the
//! multi of position `i - 1` with the embedding of token `i`.
//!
//! Weights come from `mtp.safetensors` next to the checkpoint (`scripts/qwen4_mtp_fetch.py`
//! builds it from a public conversion); `mtp.json` beside it says how they are quantized.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Array;
use mlx_rs::ops::indexing::TryIndexOp;
use serde_json::Value;

use super::{
    HyperConnection, Layer, Mixer, MoeLayer, Proj, Qsa, QsaState, Qwen4ExpCache, Qwen4ExpModel,
    owned, to_f32,
};
use crate::native_cache::NativeKvCache;
use crate::native_moe::moe_block_forward;
use crate::qwen3_5_moe::NativeWeights;
use crate::qwen4_exp::config::{QuantSpec, Qwen4ExpConfig};

/// Whether to use the draft head when its weights are present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MtpMode {
    /// Use it when the weights are present and the expert pool left after its 1.5 GB is big
    /// enough for speculation to pay (decided by the runner, which knows the pool).
    #[default]
    Auto,
    /// Use it whenever the weights are present.
    On,
    /// Never load it.
    Off,
}

/// The sidecar `mtp.json` next to `mtp.safetensors`.
#[derive(Debug, Default)]
struct MtpMeta {
    /// Norm weights already carry the `+1` (the layout the main model uses).
    norms_folded: bool,
    group_size: Option<i32>,
    /// `base -> bits`, `base` being the tensor name without `.weight`.
    bits: HashMap<String, i32>,
    /// `base -> (group_size, bits)` overrides, applied last.
    quantization: HashMap<String, QuantSpec>,
}

impl MtpMeta {
    fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("mtp.json");
        if !path.is_file() {
            return Ok(Self {
                norms_folded: true,
                ..Self::default()
            });
        }
        let v: Value = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("{}: not JSON", path.display()))?;
        let mut meta = Self {
            norms_folded: v.get("norms").and_then(Value::as_str) != Some("raw"),
            group_size: v
                .get("group_size")
                .and_then(Value::as_i64)
                .map(|g| g as i32),
            ..Self::default()
        };
        if let Some(bits) = v.get("bits").and_then(Value::as_object) {
            for (k, b) in bits {
                if let Some(b) = b.as_i64() {
                    meta.bits.insert(k.clone(), b as i32);
                }
            }
        }
        if let Some(q) = v.get("quantization").and_then(Value::as_object) {
            for (k, spec) in q {
                let g = spec.get("group_size").and_then(Value::as_i64);
                let b = spec.get("bits").and_then(Value::as_i64);
                if let (Some(g), Some(b)) = (g, b) {
                    meta.quantization.insert(
                        k.clone(),
                        QuantSpec {
                            group_size: g as i32,
                            bits: b as i32,
                        },
                    );
                }
            }
        }
        Ok(meta)
    }
}

/// The head's weights.
pub(super) struct MtpHead {
    fc_embedding: Proj,
    fc_hidden: Proj,
    pre_norm_embedding: Array,
    pre_norm_hidden: Array,
    layer: Layer,
    mixer: HyperConnection,
    /// Its routed experts are served from the expert pool like the trunk's (otherwise they
    /// are resident: 1.4 GB).
    pub(super) streamed: bool,
    /// Bytes the head keeps resident.
    pub(super) resident_bytes: u64,
}

/// What a sequence carries for the head: its attention cache and the multi stream of
/// the last consumed position, which the next entry fuses with.
#[derive(Clone)]
pub(super) struct MtpState {
    pub(super) qsa: QsaState,
    /// `[1, 1, hc*H]`; `None` before the first token has been consumed.
    pub(super) last_multi: Option<Array>,
}

impl MtpState {
    pub(super) fn new() -> Self {
        Self {
            qsa: QsaState {
                kv: NativeKvCache::new(),
                indexer_keys: None,
            },
            last_multi: None,
        }
    }

    pub(super) fn arrays(&self) -> Vec<&Array> {
        let mut out = LayerStateRef::qsa(&self.qsa);
        out.extend(self.last_multi.iter());
        out
    }
}

/// `LayerState::arrays` for a bare QSA state.
struct LayerStateRef;

impl LayerStateRef {
    fn qsa(q: &QsaState) -> Vec<&Array> {
        q.kv.keys()
            .into_iter()
            .chain(q.kv.values())
            .chain(q.indexer_keys.iter())
            .collect()
    }
}

/// Read the head's quantization off the tensors: for each `<base>.scales`, the group size
/// the sidecar names (else the checkpoint's) and the bit width the shapes imply.
fn describe_quantization(
    tensors: &HashMap<String, Array>,
    cfg: &mut Qwen4ExpConfig,
    meta: &MtpMeta,
) -> Result<()> {
    let default_gs = meta
        .group_size
        .unwrap_or(cfg.quant.default_spec().group_size);
    for (name, scales) in tensors {
        let Some(base) = name.strip_suffix(".scales") else {
            continue;
        };
        let Some(weight) = tensors.get(&format!("{base}.weight")) else {
            continue;
        };
        let spec = if let Some(spec) = meta.quantization.get(base) {
            *spec
        } else {
            let groups = i64::from(*scales.shape().last().unwrap_or(&0));
            let packed = i64::from(*weight.shape().last().unwrap_or(&0));
            let bits = match meta.bits.get(base) {
                Some(b) => i64::from(*b),
                None => {
                    // bits = packed * 32 / in_features, in_features = groups * group_size.
                    let in_features = groups * i64::from(default_gs);
                    if in_features == 0 || (packed * 32) % in_features != 0 {
                        bail!("mtp: cannot tell the bit width of {base}");
                    }
                    packed * 32 / in_features
                }
            };
            QuantSpec {
                group_size: default_gs,
                bits: bits as i32,
            }
        };
        cfg.quant.set(base, spec);
        // `MoeLayer::load_at` reads the stacked experts' spec from the parent.
        if let Some(parent) = base
            .strip_suffix(".gate_proj")
            .filter(|b| b.ends_with(".switch_mlp"))
        {
            cfg.quant.set(parent, spec);
        }
    }
    Ok(())
}

impl MtpHead {
    /// Build the head from the tensors of `mtp.safetensors` (names `mtp.*`).
    pub(super) fn load(
        tensors: HashMap<String, Array>,
        dir: &Path,
        cfg: &Qwen4ExpConfig,
        streamed: bool,
    ) -> Result<Self> {
        let meta = MtpMeta::read(dir)?;
        let mut cfg = cfg.clone();
        cfg.norms_centered = !meta.norms_folded;
        describe_quantization(&tensors, &mut cfg, &meta)?;
        // Everything resident is read now, on the loading thread (see `materialize_trunk`). A
        // streamed head's experts stay unevaluated: the pool reads them from the file.
        let resident: Vec<&Array> = tensors
            .iter()
            .filter(|(name, _)| !streamed || !name.contains(".switch_mlp."))
            .map(|(_, a)| a)
            .collect();
        for batch in resident.chunks(64) {
            mlx_rs::transforms::eval(batch.iter().copied()).context("mtp: read the head")?;
        }
        let resident_bytes = resident.iter().map(|a| a.nbytes() as u64).sum();
        let w = NativeWeights::from_tensors(tensors);
        let layer = Layer {
            attn_hc: HyperConnection::load(&w, &cfg, "mtp.layers.0.attn_hyper_connection", true)?,
            mlp_hc: HyperConnection::load(&w, &cfg, "mtp.layers.0.mlp_hyper_connection", true)?,
            mixer: Mixer::Qsa(Qsa::load(&w, &cfg, "mtp.layers.0.self_attn")?),
            moe: MoeLayer::load_at(
                &w,
                &cfg,
                "mtp.layers.0.mlp",
                if streamed {
                    crate::offload::MTP_LAYER
                } else {
                    0
                },
            )?,
            ple: None,
        };
        Ok(Self {
            fc_embedding: Proj::load(&w, &cfg, "mtp.fc_embedding")?,
            fc_hidden: Proj::load(&w, &cfg, "mtp.fc_hidden")?,
            pre_norm_embedding: super::centered_norm(&w, &cfg, "mtp.pre_fc_norm_embedding.weight")?,
            pre_norm_hidden: super::centered_norm(&w, &cfg, "mtp.pre_fc_norm_hidden.weight")?,
            layer,
            mixer: HyperConnection::load(&w, &cfg, "mtp.hyper_connection_mixer", false)?,
            streamed,
            resident_bytes,
        })
    }
}

impl Qwen4ExpModel {
    /// Is the draft head loaded?
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Bytes the head keeps resident (0 without one).
    pub fn mtp_resident_bytes(&self) -> u64 {
        self.mtp.as_ref().map_or(0, |h| h.resident_bytes)
    }

    fn head(&self) -> Result<&MtpHead> {
        self.mtp
            .as_ref()
            .ok_or_else(|| anyhow!("qwen4_exp: no MTP draft head is loaded"))
    }

    /// `fuse` of the reference: `[1, S, H]` embeddings and `[1, S, hc*H]` multi streams to
    /// the `[1, S, hc*H]` the head's layer takes.
    fn mtp_fuse(&self, head: &MtpHead, embedded: &Array, multi: &Array) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, s) = (embedded.shape()[0], embedded.shape()[1]);
        let (hc, d) = (cfg.hc_count as i32, cfg.hidden_size as i32);
        let e = head.fc_embedding.apply(&mlx_rs::fast::rms_norm(
            embedded,
            &head.pre_norm_embedding,
            cfg.rms_norm_eps,
        )?)?;
        let h = mlx_rs::fast::rms_norm(multi, &head.pre_norm_hidden, cfg.rms_norm_eps)?;
        let h = mlx_rs::ops::reshape(&h, &[b, s, hc, d])?;
        let h = head.fc_hidden.apply(&h)?;
        let h = mlx_rs::ops::add(&mlx_rs::ops::expand_dims(&e, -2)?, &h)?;
        Ok(mlx_rs::ops::reshape(&h, &[b, s, hc * d])?)
    }

    /// One pass of the head over `S` already-embedded tokens: returns the `[1, S, H]` the
    /// shared lm_head reads and the `[1, S, hc*H]` multi it leaves.
    fn mtp_step(
        &self,
        head: &MtpHead,
        embedded: &Array,
        multi: &Array,
        st: &mut QsaState,
    ) -> Result<(Array, Array)> {
        let Mixer::Qsa(qsa) = &head.layer.mixer else {
            bail!("qwen4_exp: the MTP head's layer is not a full-attention layer");
        };
        let h = self.mtp_fuse(head, embedded, multi)?;
        let (x, inject) = self.hyper_mix(&head.layer.attn_hc, &h)?;
        let inject = inject.ok_or_else(|| anyhow!("mtp: attn hyper-connection has no inject"))?;
        let x = self.qsa_forward(qsa, &x, st, super::RopeAt::Index)?;
        let h = self.write_back(&h, &x, &inject)?;
        let (x, inject) = self.hyper_mix(&head.layer.mlp_hc, &h)?;
        let inject = inject.ok_or_else(|| anyhow!("mtp: mlp hyper-connection has no inject"))?;
        let offload = if head.streamed {
            self.offload.as_ref()
        } else {
            None
        };
        let x = moe_block_forward(&x, &head.layer.moe.view(offload)).context("mtp: moe block")?;
        let h = self.write_back(&h, &x, &inject)?;
        let (mixed, _) = self.hyper_mix(&head.mixer, &h)?;
        Ok((mixed, h))
    }

    /// The attention half of a head pass: all that is needed to write the entries of tokens
    /// whose drafts nobody reads (the prompt, and the tokens a verify pass accepted).
    fn mtp_consume_rows(
        &self,
        head: &MtpHead,
        embedded: &Array,
        multi: &Array,
        st: &mut QsaState,
    ) -> Result<()> {
        let Mixer::Qsa(qsa) = &head.layer.mixer else {
            bail!("qwen4_exp: the MTP head's layer is not a full-attention layer");
        };
        let h = self.mtp_fuse(head, embedded, multi)?;
        let (x, _) = self.hyper_mix(&head.layer.attn_hc, &h)?;
        let x = self.qsa_forward(qsa, &x, st, super::RopeAt::Index)?;
        // Nothing reads the attention output, but its cache writes must be real: settle them
        // now so a lazy graph over a whole prompt chunk does not outlive the chunk.
        let mut pending = LayerStateRef::qsa(st);
        pending.push(&x);
        mlx_rs::transforms::eval(pending).context("mtp: settle the head's cache")?;
        Ok(())
    }

    /// Rows `[1, S, H]` of the embedding table for `ids`, in f32.
    pub(super) fn embed_rows(&self, ids: &[u32]) -> Result<Array> {
        let e = self.embed_untiled(ids)?;
        to_f32(e)
    }

    /// Feed the tokens `ids` (whose final pre-mixer residual streams are `multi`, `[1, S,
    /// hc*H]`) through the head so its cache stays one entry per consumed token but the first.
    /// `first` says these are the first tokens of the sequence (no earlier multi to fuse with).
    pub(super) fn mtp_consume(
        &self,
        st: &mut MtpState,
        ids: &[u32],
        multi: &Array,
        first: bool,
    ) -> Result<()> {
        let head = self.head()?;
        let s = ids.len() as i32;
        let last = multi.try_index((.., (s - 1)..s, ..))?;
        let last = owned(&last)?;
        let (rows, multis) = match (st.last_multi.take(), first) {
            (Some(prev), false) => {
                let multis = if s > 1 {
                    mlx_rs::ops::concatenate_axis(
                        &[&prev, &multi.try_index((.., ..(s - 1), ..))?],
                        1,
                    )?
                } else {
                    prev
                };
                (ids, Some(multis))
            }
            // The first token of a sequence has no earlier position to fuse with.
            (_, true) if s > 1 => (&ids[1..], Some(multi.try_index((.., ..(s - 1), ..))?)),
            (_, true) => (&ids[1..], None),
            (None, false) => bail!("mtp: no earlier multi stream to continue from"),
        };
        if let Some(multis) = multis
            && !rows.is_empty()
        {
            let embedded = self.embed_rows(rows)?;
            self.mtp_consume_rows(head, &embedded, &multis, &mut st.qsa)?;
        }
        st.last_multi = Some(last);
        Ok(())
    }

    /// The head's guess for the token after the one `token` is followed by: fuses `token`'s
    /// embedding with the multi stream of the position before it, appends that entry to the
    /// head's cache (it is the right entry for `token` whatever the verifier says), and
    /// returns the argmax of the shared lm_head over the head's output.
    pub(super) fn mtp_draft(&self, st: &mut MtpState, token: u32) -> Result<u32> {
        let head = self.head()?;
        let multi = st
            .last_multi
            .as_ref()
            .ok_or_else(|| anyhow!("mtp: draft before any position was consumed"))?;
        let embedded = self.embed_rows(&[token])?;
        let (mixed, _) = self.mtp_step(head, &embedded, multi, &mut st.qsa)?;
        let lm = self
            .lm_head
            .as_ref()
            .ok_or_else(|| anyhow!("qwen4_exp: tied embeddings are not supported"))?;
        let logits = to_f32(lm.apply(&mixed)?)?;
        let flat = mlx_rs::ops::reshape(&logits, &[-1])?;
        let idx = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false)?;
        idx.eval().context("mtp: draft argmax")?;
        // The entry just written must be real before the verify pass builds on it.
        let mut pending = LayerStateRef::qsa(&st.qsa);
        pending.extend(st.last_multi.iter());
        mlx_rs::transforms::eval(pending).context("mtp: settle the draft entry")?;
        Ok(idx.item::<u32>())
    }

    /// The head's guess for the token that follows `token`'s successor, made before the model
    /// has processed `token` (see [`Self::mtp_draft`]); `None` when this sequence's cache has
    /// no usable head state.
    pub(crate) fn draft_for(&self, cache: &mut Qwen4ExpCache, token: u32) -> Result<Option<u32>> {
        let Some(st) = cache.mtp.as_mut() else {
            return Ok(None);
        };
        if st.last_multi.is_none() {
            return Ok(None);
        }
        self.mtp_draft(st, token).map(Some)
    }

    /// Does this sequence carry a usable draft-head state?
    pub(crate) fn mtp_ready(&self, cache: &Qwen4ExpCache) -> bool {
        self.mtp.is_some() && cache.mtp.as_ref().is_some_and(|m| m.last_multi.is_some())
    }

    /// Settle the head's state after a verify pass over `[pending, draft]`:
    /// the first-row state `first` gets the multi of the pending token's position; when the
    /// draft was accepted, `full` (both rows) also gets the draft's entry and the multi after it.
    pub(crate) fn mtp_finish_round(
        &self,
        first: &mut Qwen4ExpCache,
        full: Option<&mut Qwen4ExpCache>,
        multi: &Array,
        draft: u32,
    ) -> Result<()> {
        let head = self.head()?;
        let row0 = owned(&multi.try_index((.., 0..1, ..))?)?;
        if let Some(st) = first.mtp.as_mut() {
            st.last_multi = Some(row0.clone());
        }
        if let Some(full) = full
            && let Some(st) = full.mtp.as_mut()
        {
            let embedded = self.embed_rows(&[draft])?;
            self.mtp_consume_rows(head, &embedded, &row0, &mut st.qsa)?;
            st.last_multi = Some(owned(&multi.try_index((.., 1..2, ..))?)?);
        }
        Ok(())
    }

    /// The head over `ids` and `multi` (`[1, S, hc*H]`, one row per id) fed in chunks of the
    /// given lengths into a fresh cache: the `[1, len, vocab]` logits of each chunk.
    #[cfg(test)]
    pub(crate) fn mtp_logits_for_test(
        &self,
        ids: &[u32],
        multi: &Array,
        chunks: &[usize],
    ) -> Result<Vec<Array>> {
        let head = self.head()?;
        let mut st = MtpState::new();
        let lm = self.lm_head.as_ref().ok_or_else(|| anyhow!("no lm_head"))?;
        let mut at = 0;
        let mut out = Vec::new();
        for &len in chunks {
            let embedded = self.embed_rows(&ids[at..at + len])?;
            let rows = multi.try_index((.., at as i32..(at + len) as i32, ..))?;
            let (mixed, _) = self.mtp_step(head, &embedded, &rows, &mut st.qsa)?;
            let logits = to_f32(lm.apply(&mixed)?)?;
            logits.eval()?;
            out.push(logits);
            at += len;
        }
        Ok(out)
    }

    /// Entries in the head's cache (0 without a head state).
    #[cfg(test)]
    pub(crate) fn mtp_entries(&self, cache: &Qwen4ExpCache) -> usize {
        cache.mtp.as_ref().map_or(0, |m| m.qsa.kv.offset())
    }
}
