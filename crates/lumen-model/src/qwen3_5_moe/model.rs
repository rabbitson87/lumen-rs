//! The top-level Qwen3.5-VL-MoE text model: 40 decoder layers + embed + final norm + lm_head.
//!
//! This module owns the model-level composition, not the weight materialization. A future
//! Stage 2-f-b loader will take a `Classification` (from [`super::weights`]) plus the 4-shard
//! safetensors files and return a fully populated [`Qwen3_5MoeTextModel`]. Until then the
//! tests here use synthetic tiny weights to lock down:
//!   - the 40-layer dispatch pattern (3 linear + 1 full × 10 cycles),
//!   - embedding → decoder stack → final norm → lm_head wiring,
//!   - the prefill vs decode entry-point (`forward(input_ids)` returns `[B, S, vocab]`).
//!
//! Reference: `mlx_lm.models.qwen3_5.Qwen3_5TextModel` + `TextModel` (mlx-lm 0.31.3).
//!
//! ## Not yet wired (Stage 2-f-b / 2-f-c follow-ups)
//!   - Causal / SSM masks — we pass `None` through the forward; the per-block fixtures
//!     already validate that this matches MLX's `cache=None, N>1` dispatch (self-attn gets
//!     `"causal"` as a string constant, SSM gets `None`). When we land real decoding, a mask
//!     helper will materialize the causal tensor once per prefill and route it.
//!   - KV cache. Stage 5 integrates TurboQuant there.
//!   - Tied word embeddings. The shipped checkpoint has `tie_word_embeddings=false`, so we
//!     always require an `lm_head`. If a future checkpoint flips the flag, `lm_head` can be
//!     constructed from `embed_tokens.embeddings()` via `Linear::new(weight.clone(), None)`.

use candle_core::{DType, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, RmsNorm};

use super::layer::{AttentionBlockSnapshot, CompressedKvHandle, DecoderLayer};
use super::proj::ProjLinear;

/// Snapshot of every layer's recurrent / append-cache state. Returned by
/// [`Qwen3_5MoeTextModel::snapshot_state`] and consumed by
/// [`Qwen3_5MoeTextModel::restore_state`]. Used by speculative decoding to
/// roll back after a verify-batch forward.
#[derive(Clone)]
pub struct Qwen3_5MoeStateSnapshot {
    layers: Vec<AttentionBlockSnapshot>,
}

impl Qwen3_5MoeStateSnapshot {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

pub(crate) fn dump_tensor_f32_public(t: &Tensor, path: &str) -> CandleResult<()> {
    dump_tensor_f32(t, path)
}

/// Serialize a tensor's contents as little-endian f32 bytes at `path`. Writes a tiny header:
/// `b"TQHD"` + rank(u32) + dims(u32 each) + data.
fn dump_tensor_f32(t: &Tensor, path: &str) -> CandleResult<()> {
    use std::io::Write;
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let dims = t.dims().to_vec();
    let mut f = std::fs::File::create(path)
        .map_err(|e| candle_core::Error::Msg(format!("create {path}: {e}")))?;
    f.write_all(b"TQHD")
        .map_err(|e| candle_core::Error::Msg(format!("write magic: {e}")))?;
    f.write_all(&(dims.len() as u32).to_le_bytes())
        .map_err(|e| candle_core::Error::Msg(format!("write rank: {e}")))?;
    for d in &dims {
        f.write_all(&(*d as u32).to_le_bytes())
            .map_err(|e| candle_core::Error::Msg(format!("write dim: {e}")))?;
    }
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
    f.write_all(bytes)
        .map_err(|e| candle_core::Error::Msg(format!("write data: {e}")))?;
    Ok(())
}

/// Fully composed Qwen3.5-VL-MoE text model (no vision tower). Construct via
/// [`Qwen3_5MoeTextModel::new`]; the caller wires the [`Embedding`], 40 [`DecoderLayer`]s,
/// the final [`RmsNorm`], and the `lm_head` [`Linear`].
pub struct Qwen3_5MoeTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: ProjLinear,
    /// Optional TurboQuant-compressed KV cache backend, shared across all full-attention
    /// layers. Populated by `enable_turboquant`; consumed inside each
    /// `SelfAttention::forward_with_tq` via the `tq_slot` assigned by the loader.
    compressed_kv: CompressedKvHandle,
}

impl Qwen3_5MoeTextModel {
    pub fn new(
        embed_tokens: Embedding,
        layers: Vec<DecoderLayer>,
        final_norm: RmsNorm,
        lm_head: ProjLinear,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            compressed_kv: None,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    /// Prefill forward. `input_ids: [B, S]` (token IDs). Returns logits `[B, S, vocab]`.
    ///
    /// The forward currently passes `None` for both attention masks. This matches the MLX
    /// reference behavior when `cache=None` for a prefill step whose S>1: `create_ssm_mask`
    /// returns `None` unconditionally, and `create_attention_mask` returns the `"causal"`
    /// sentinel which `scaled_dot_product_attention` interprets as a causal mask internally.
    /// The Candle self-attn port here doesn't have that "causal" sentinel yet, so for now
    /// the block-level fixture tests exercise the masked path explicitly via
    /// `SelfAttention::prefill_causal_mask(...)`. End-to-end masked prefill lands in
    /// Stage 2-f-b once the loader is in.
    pub fn forward(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_offset(input_ids, 0)
    }

    /// Forward with an explicit position offset for RoPE + KV cache continuation.
    ///
    /// `seqlen_offset`: index of the first token in `input_ids` relative to the start of
    /// the full sequence. `0` for prefill; `past_seq_len` on each autoregressive decode call.
    /// The caller is responsible for `reset_cache()`-ing between independent requests.
    pub fn forward_with_offset(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> CandleResult<Tensor> {
        // Destructure so `layers` and `compressed_kv` are borrowed separately.
        let Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            compressed_kv,
        } = self;
        // Optional layer-wise hidden-state dump. Set `LUMEN_DUMP_HIDDEN=/path/to/dir`
        // before running a prefill and every intermediate [B, S, hidden] / [B, S, vocab]
        // tensor is serialized as raw f32 to that directory for diffing against MLX.
        let dump_dir = std::env::var("LUMEN_DUMP_HIDDEN").ok();
        // Optional forward-level breakdown: `LUMEN_FORWARD_BREAKDOWN=1` prints
        // embed / layers / final_norm / lm_head ms per call. Each marker syncs the
        // device, so off by default. Used to size the missing-time contributors
        // (embed lookup vs lm_head matmul vs sampling) when LUMEN_LAYER_TIMING
        // alone doesn't account for the full token decode wall time.
        let breakdown = std::env::var("LUMEN_FORWARD_BREAKDOWN")
            .map(|v| v == "1")
            .unwrap_or(false);
        let device = input_ids.device().clone();
        let t_start = if breakdown {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut h = embed_tokens.forward(input_ids)?;
        if let Some(d) = &dump_dir {
            dump_tensor_f32(&h, &format!("{d}/embed.bin"))?;
        }
        // Workstream B Phase 10 (2026-05-09): model-wide bf16 carrier.
        // When `LUMEN_BF16_RESIDUAL=1` (and the chain prerequisite
        // `LUMEN_BF16_RMSNORM=1` is also set), cast the embedding output
        // to bf16 ONCE here. All 64 layers run with bf16 carrier; the
        // boundary cast back to f32 happens once before `final_norm` /
        // `lm_head`. Total cast: 2/token vs Phase 9's 192/token.
        //
        // Layer self-gating: each `DecoderLayer::forward_with_tq` checks the
        // same env-flag chain in its `bf16_residual_active` gate. When that
        // gate is true, the layer's input_layernorm + post_attention_layernorm
        // use the bf16-in shaders (`apply_rms_norm_bf16_in_bf16_out`); o_proj /
        // out_proj keep bf16 (boundary lifts from B.9); the residual adds
        // run bf16+bf16. When false, the legacy f32 stream fires unchanged.
        //
        // Mutual exclusion with `LUMEN_LEVER_L4=1` (which threads pre-
        // normed f32 attn_in across layers) — the layer-level gate already
        // disables B.9/B.10 when prev_attn_in is Some, but skipping the
        // model-entry cast is cheaper.
        let bf16_carrier_active = std::env::var("LUMEN_BF16_RESIDUAL")
            .map(|v| v == "1")
            .unwrap_or(false)
            && std::env::var("LUMEN_BF16_RMSNORM")
                .map(|v| v == "1")
                .unwrap_or(false)
            && device.is_metal()
            && std::env::var("LUMEN_LEVER_L4").as_deref() != Ok("1");
        if bf16_carrier_active {
            h = h.to_dtype(candle_core::DType::BF16)?;
        }
        let t_embed = if breakdown {
            let _ = device.synchronize();
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Lever L4: when LUMEN_LEVER_L4=1 (default OFF until σ ≥ +2 confirmed),
        // each layer's mlp_final fused kernel ALSO produces the next layer's
        // pre-normalized attn_in (using next layer's input_layernorm weight),
        // saving one input_layernorm dispatch per layer transition. layer 0
        // still pays input_layernorm; layer N-1 produces None (no next layer).
        let lever_l4_active = std::env::var("LUMEN_LEVER_L4").as_deref() == Ok("1");
        // Pre-extract per-layer input_layernorm weights/eps for L4 carry —
        // avoids holding two mutable borrows of `layers` simultaneously.
        let next_rms_weights: Vec<Tensor> = if lever_l4_active {
            layers
                .iter()
                .map(|l| l.input_layernorm.weight().clone())
                .collect()
        } else {
            Vec::new()
        };
        let next_rms_eps: Vec<f32> = if lever_l4_active {
            layers
                .iter()
                .map(|l| l.input_layernorm.eps() as f32)
                .collect()
        } else {
            Vec::new()
        };
        let mut prev_attn_in: Option<Tensor> = None;
        let n_layers = layers.len();
        for (i, layer) in layers.iter_mut().enumerate() {
            let next_rms: Option<(&Tensor, f32)> = if lever_l4_active && i + 1 < n_layers {
                Some((&next_rms_weights[i + 1], next_rms_eps[i + 1]))
            } else {
                None
            };
            let (h_new, next_attn_in) = layer.forward_with_tq(
                &h,
                seqlen_offset,
                None,
                None,
                compressed_kv,
                prev_attn_in.as_ref(),
                next_rms,
            )?;
            h = h_new;
            prev_attn_in = next_attn_in;
            if let Some(d) = &dump_dir {
                dump_tensor_f32(&h, &format!("{d}/L{i:02}.bin"))?;
            }
        }
        let t_layers = if breakdown {
            let _ = device.synchronize();
            Some(std::time::Instant::now())
        } else {
            None
        };
        // cast bf16 carrier back to f32 once before
        // `final_norm` + `lm_head` (these stay on the f32 path — neither
        // has a bf16-in fast path yet). No-op when bf16_carrier_active was
        // false (the chain already runs f32).
        let h = if bf16_carrier_active && h.dtype() != candle_core::DType::F32 {
            h.to_dtype(candle_core::DType::F32)?
        } else {
            h
        };
        let h = {
            #[cfg(feature = "mpsgraph")]
            {
                if let Some(m) = super::mpsgraph_norm::get() {
                    m.forward(&h, final_norm.weight())?
                } else {
                    final_norm.forward(&h)?
                }
            }
            #[cfg(not(feature = "mpsgraph"))]
            {
                final_norm.forward(&h)?
            }
        };
        if let Some(d) = &dump_dir {
            dump_tensor_f32(&h, &format!("{d}/final_norm.bin"))?;
        }
        let t_norm = if breakdown {
            let _ = device.synchronize();
            Some(std::time::Instant::now())
        } else {
            None
        };
        let logits = lm_head.forward(&h)?;
        if let Some(d) = &dump_dir {
            dump_tensor_f32(&logits, &format!("{d}/logits.bin"))?;
        }
        if breakdown {
            let _ = device.synchronize();
            let t_end = std::time::Instant::now();
            let t0 = t_start.unwrap();
            let te = t_embed.unwrap();
            let tl = t_layers.unwrap();
            let tn = t_norm.unwrap();
            let embed_ms = te.duration_since(t0).as_secs_f64() * 1000.0;
            let layers_ms = tl.duration_since(te).as_secs_f64() * 1000.0;
            let norm_ms = tn.duration_since(tl).as_secs_f64() * 1000.0;
            let head_ms = t_end.duration_since(tn).as_secs_f64() * 1000.0;
            let total_ms = t_end.duration_since(t0).as_secs_f64() * 1000.0;
            let vocab = logits.dims().last().copied().unwrap_or(0);
            eprintln!(
                "    fwd-breakdown: embed={embed_ms:.1} layers={layers_ms:.1} \
                 norm={norm_ms:.1} head={head_ms:.1} total={total_ms:.1}ms (vocab={vocab})"
            );
        }
        Ok(logits)
    }

    /// Attach a compressed-KV backend (e.g. TurboQuant `GpuCompressor`). The backend's layer
    /// indexing must match each full-attention layer's `tq_slot` as assigned by the loader.
    pub fn set_compressed_kv(
        &mut self,
        backend: Box<dyn candle_transformers::models::quantized_gemma4::CompressedKVBackend + Send>,
    ) {
        self.compressed_kv = Some(backend);
    }

    pub fn has_compressed_kv(&self) -> bool {
        self.compressed_kv.is_some()
    }

    pub fn clear_compressed_kv(&mut self) {
        if let Some(ckv) = self.compressed_kv.as_mut() {
            ckv.clear();
        }
    }

    /// Enable a vanilla `KvCache` on every full-attention layer (10 of 40 for Qwen3.5-MoE).
    /// Call once after loading; subsequent `forward_with_offset` calls will accumulate K/V
    /// across tokens.
    pub fn enable_kv_cache(&mut self, max_seq_len: usize) {
        for layer in self.layers.iter_mut() {
            layer.enable_kv_cache(max_seq_len);
        }
    }

    /// Reset every layer's KV cache (start of a new request). Also clears the compressed
    /// backend if one is attached.
    pub fn reset_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_cache();
        }
        self.clear_compressed_kv();
    }

    /// Set the active sequence ID across all full-attention layers.
    /// Linear-attention (SSM) layers are unaffected — Phase 1 limitation.
    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        for layer in self.layers.iter_mut() {
            layer.set_current_seq_id(seq_id);
        }
    }

    /// Allocate per-seq KV cache slots across all full-attention layers.
    /// Call before the first prefill forward for a new `seq_id`.
    /// `reset_cache()` must NOT be called after this — it would wipe all seqs.
    pub fn init_sequence(&mut self, seq_id: u64) {
        for layer in self.layers.iter_mut() {
            layer.init_seq_kv_cache(seq_id);
        }
    }

    /// Free KV cache memory for a completed sequence.
    pub fn remove_sequence(&mut self, seq_id: u64) {
        for layer in self.layers.iter_mut() {
            layer.remove_seq_kv_cache(seq_id);
        }
    }

    /// B sequential forward passes with per-seq SSM state isolation.
    /// Each sequence's SSM (conv + recurrent state) is swapped in/out via the lazy-swap
    /// pattern in `GatedDeltaNet::set_current_seq_id`, so linear-attention state is no
    /// longer shared across concurrent sequences.
    pub fn forward_batch_decode_seqs(
        &mut self,
        last_tokens: &[u32],
        seq_ids: &[u64],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        let device = self.embed_tokens.embeddings().device().clone();
        let mut rows: Vec<Tensor> = Vec::with_capacity(seq_ids.len());
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            self.set_current_seq_id(seq_id);
            let tok = Tensor::new(&[last_tokens[i]], &device)?.unsqueeze(0)?; // [1, 1]
            let logits = self.forward_with_offset(&tok, positions[i])?; // [1, 1, vocab]
            rows.push(logits.squeeze(0)?.squeeze(0)?); // [vocab]
        }
        Tensor::stack(&rows, 0) // [B, vocab]
    }

    /// CB Phase 0 diagnostic: per-seq per-layer forward (like v1) with
    /// batched final_norm + lm_head only. Isolates the per-layer `Tensor::cat`
    /// overhead and `MoE bl=2` inefficiency from the rest of v2's path.
    ///
    /// If `phase0 ≈ v1`, the v2 regression is dominated by per-layer cat +
    /// `bl=2` MoE work; a kernel-level multi-token MoE rework is the main lever.
    /// If `phase0 ≈ v2`, the cat itself is the dominant cost.
    pub fn forward_batch_decode_seqs_phase0(
        &mut self,
        last_tokens: &[u32],
        seq_ids: &[u64],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        let n = seq_ids.len();
        let Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            compressed_kv,
        } = self;

        let device = embed_tokens.embeddings().device().clone();

        let mut hs: Vec<Tensor> = (0..n)
            .map(|i| {
                let tok = Tensor::new(&[last_tokens[i]], &device)?;
                embed_tokens.forward(&tok)?.unsqueeze(0) // [1, 1, hidden]
            })
            .collect::<CandleResult<Vec<_>>>()?;

        // Per layer: per-seq forward (no cat). Each seq's full layer pass
        // (attn + MoE residuals) runs at bl=1 — same kernel path as v1.
        for layer in layers.iter_mut() {
            for (i, &seq_id) in seq_ids.iter().enumerate() {
                layer.set_current_seq_id(seq_id);
                let h_in = hs[i].clone(); // Arc bump — cheap on Metal
                let (h_out, _) = layer.forward_with_tq(
                    &h_in,
                    positions[i],
                    None,
                    None,
                    compressed_kv,
                    None, // L4 not active in batched path
                    None,
                )?;
                hs[i] = h_out;
            }
        }

        // Final norm + lm_head: batched once across all seqs
        let h_stacked = Tensor::cat(&hs, 0)?; // [N, 1, hidden]
        let h_normed = {
            #[cfg(feature = "mpsgraph")]
            {
                if let Some(m) = super::mpsgraph_norm::get() {
                    m.forward(&h_stacked, final_norm.weight())?
                } else {
                    final_norm.forward(&h_stacked)?
                }
            }
            #[cfg(not(feature = "mpsgraph"))]
            {
                final_norm.forward(&h_stacked)?
            }
        };
        let logits = lm_head.forward(&h_normed)?; // [N, 1, vocab]
        logits.squeeze(1) // [N, vocab]
    }

    /// CB Phase 2: batched MoE decode.
    ///
    /// Per-layer: sequential attention (SSM state isolation preserved via
    /// `set_current_seq_id`) then a single batched MoE call on the stacked
    /// `[B, 1, hidden]` hiddens. Final norm + lm_head also run once on the
    /// whole batch. Numerically equivalent to `forward_batch_decode_seqs`
    /// since MoE routing and expert matmuls are independent across tokens.
    ///
    /// `LUMEN_V2_BREAKDOWN=1` prints per-stage ms (embed / attn / cat / moe / split / head).
    pub fn forward_batch_decode_seqs_v2(
        &mut self,
        last_tokens: &[u32],
        seq_ids: &[u64],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        let n = seq_ids.len();
        let breakdown = std::env::var("LUMEN_V2_BREAKDOWN")
            .map(|v| v == "1")
            .unwrap_or(false);
        let Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            compressed_kv,
        } = self;

        let device = embed_tokens.embeddings().device().clone();
        let mark = |on: bool, dev: &candle_core::Device| -> Option<std::time::Instant> {
            if on {
                let _ = dev.synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            }
        };
        let t0 = mark(breakdown, &device);

        // Embed each token: Vec<[1, 1, hidden]>
        let mut hs: Vec<Tensor> = (0..n)
            .map(|i| {
                let tok = Tensor::new(&[last_tokens[i]], &device)?;
                embed_tokens.forward(&tok)?.unsqueeze(0) // [1, 1, hidden]
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let t_embed = mark(breakdown, &device);

        let mut attn_ms = 0.0f64;
        let mut cat_ms = 0.0f64;
        let mut moe_ms = 0.0f64;
        let mut split_ms = 0.0f64;

        // Per layer: sequential attention (SSM state isolation) + batched MoE
        for layer in layers.iter_mut() {
            let ta = mark(breakdown, &device);
            let mut post_attn: Vec<Tensor> = Vec::with_capacity(n);
            for (i, &seq_id) in seq_ids.iter().enumerate() {
                layer.set_current_seq_id(seq_id);
                let h_out =
                    layer.forward_attn_part(&hs[i], positions[i], None, None, compressed_kv)?;
                post_attn.push(h_out);
            }
            let tc = mark(breakdown, &device);
            // One MoE call for all B sequences — routing is per-token independent
            let h_batch = Tensor::cat(&post_attn, 0)?; // [n, 1, hidden]
            let tm = mark(breakdown, &device);
            let out_batch = layer.forward_moe_part(&h_batch)?; // [n, 1, hidden]
            let ts = mark(breakdown, &device);
            hs = (0..n)
                .map(|i| out_batch.i(i..(i + 1)))
                .collect::<CandleResult<Vec<_>>>()?;
            let te = mark(breakdown, &device);
            if breakdown {
                attn_ms += tc.unwrap().duration_since(ta.unwrap()).as_secs_f64() * 1000.0;
                cat_ms += tm.unwrap().duration_since(tc.unwrap()).as_secs_f64() * 1000.0;
                moe_ms += ts.unwrap().duration_since(tm.unwrap()).as_secs_f64() * 1000.0;
                split_ms += te.unwrap().duration_since(ts.unwrap()).as_secs_f64() * 1000.0;
            }
        }
        let t_layers = mark(breakdown, &device);

        // Batched final norm + lm_head (one dispatch for all B seqs)
        let h_stacked = Tensor::cat(&hs, 0)?; // [n, 1, hidden]
        let h_normed = {
            #[cfg(feature = "mpsgraph")]
            {
                if let Some(m) = super::mpsgraph_norm::get() {
                    m.forward(&h_stacked, final_norm.weight())?
                } else {
                    final_norm.forward(&h_stacked)?
                }
            }
            #[cfg(not(feature = "mpsgraph"))]
            {
                final_norm.forward(&h_stacked)?
            }
        };
        let logits = lm_head.forward(&h_normed)?; // [n, 1, vocab]
        let out = logits.squeeze(1)?; // [n, vocab]
        if breakdown {
            let _ = device.synchronize();
            let t_end = std::time::Instant::now();
            let embed_ms = t_embed.unwrap().duration_since(t0.unwrap()).as_secs_f64() * 1000.0;
            let layers_ms = t_layers
                .unwrap()
                .duration_since(t_embed.unwrap())
                .as_secs_f64()
                * 1000.0;
            let head_ms = t_end.duration_since(t_layers.unwrap()).as_secs_f64() * 1000.0;
            eprintln!(
                "    v2-bd: embed={embed_ms:.1} layers={layers_ms:.1} \
                 (attn={attn_ms:.1} cat={cat_ms:.1} moe={moe_ms:.1} split={split_ms:.1}) \
                 head={head_ms:.1}"
            );
        }
        Ok(out)
    }

    /// Capture all per-layer recurrent / append-cache state for speculative
    /// decoding rollback. The returned snapshot is independent of further
    /// forward calls — restoring it is sufficient to undo any K-token
    /// extension applied after the snapshot was taken.
    pub fn snapshot_state(&self) -> CandleResult<Qwen3_5MoeStateSnapshot> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.snapshot_state())
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Qwen3_5MoeStateSnapshot { layers })
    }

    /// Restore all layer states from a snapshot produced by
    /// [`snapshot_state`]. Truncates compressed-KV slots in lockstep so the
    /// shared TurboQuant backend stays consistent.
    pub fn restore_state(&mut self, snap: &Qwen3_5MoeStateSnapshot) -> CandleResult<()> {
        if snap.layers.len() != self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "restore_state: layer count mismatch (snap={}, model={})",
                snap.layers.len(),
                self.layers.len()
            )));
        }
        for (layer, layer_snap) in self.layers.iter_mut().zip(snap.layers.iter()) {
            layer.restore_state(layer_snap, &mut self.compressed_kv)?;
        }
        Ok(())
    }

    /// Walk the layer stack and assign each full-attention layer a sparse TurboQuant slot
    /// (0..num_full_attn). Linear-attn layers receive `None`. Must be called after loading
    /// the layers (the loader already sets `layer_idx` via `set_layer_idx`).
    pub fn assign_tq_slots(&mut self) -> usize {
        use super::layer::AttentionBlock;
        let mut slot = 0;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            layer.set_layer_idx(idx);
            if let AttentionBlock::Full(sa) = layer.attention_mut() {
                sa.set_layer_idx(idx);
                sa.set_tq_slot(Some(slot));
                slot += 1;
            }
        }
        slot
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::config::{LayerType, Qwen3_5MoeConfig};
    use crate::qwen3_5_moe::layer::{AttentionBlock, DecoderLayer};
    use crate::qwen3_5_moe::linear_attn::{
        GatedDeltaNet, GatedDeltaNetRuntime, LinearAttnDims, conv1d_from_mlx_weight,
    };
    use crate::qwen3_5_moe::moe::{
        MoeDims, SharedExpert, SparseMoeBlock, SparseMoeRuntime, SwitchMlp,
    };
    use crate::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

    use candle_core::{Device, Tensor};
    use candle_nn::{Embedding, Linear, RmsNorm};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const CONFIG_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_5_moe_config.json"
    ));

    // Tiny dims so the full 40-layer synthetic model stays well under a second on CPU.
    const HIDDEN: usize = 16;
    const VOCAB: usize = 32;

    fn rnd(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.05..0.05)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn tiny_self_attn(rng: &mut StdRng, device: &Device) -> SelfAttention {
        let d = SelfAttnDims {
            hidden_size: HIDDEN,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            attn_output_gate: true,
            rotary_dim: 4,
        };
        // Option M2: pre-fused [q_out + 2*kv_out, hidden] qkv weight.
        let combined_out = d.q_out_dim() + 2 * d.kv_out_dim();
        let qkv = Linear::new(rnd(&[combined_out, d.hidden_size], rng, device), None);
        let o = Linear::new(rnd(&[d.hidden_size, d.attn_value_dim()], rng, device), None);
        let ones = Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        SelfAttention::new(
            SelfAttnRuntime {
                dims: d,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-6,
            },
            qkv.into(),
            o.into(),
            RmsNorm::new(ones.clone(), 1e-6),
            RmsNorm::new(ones, 1e-6),
        )
    }

    fn tiny_linear_attn(rng: &mut StdRng, device: &Device) -> GatedDeltaNet {
        let d = LinearAttnDims {
            hidden_size: HIDDEN,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 4,
            conv_kernel: 4,
        };
        // Option M: pre-fused [qkv_dim + v_dim + 2*Hv, hidden] in_proj.
        let combined_out = d.qkv_dim() + d.v_dim() + 2 * d.num_v_heads;
        let in_proj = Linear::new(rnd(&[combined_out, d.hidden_size], rng, device), None);
        let conv_w = rnd(&[d.qkv_dim(), d.conv_kernel, 1], rng, device);
        let conv = conv1d_from_mlx_weight(conv_w, d.conv_kernel).unwrap();
        let a_log = rnd(&[d.num_v_heads], rng, device);
        let dt_bias = rnd(&[d.num_v_heads], rng, device);
        let norm_w = Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        let out = Linear::new(rnd(&[d.hidden_size, d.v_dim()], rng, device), None);
        GatedDeltaNet::new(
            GatedDeltaNetRuntime {
                dims: d,
                rms_norm_eps: 1e-6,
            },
            in_proj.into(),
            conv,
            a_log,
            dt_bias,
            norm_w,
            out.into(),
        )
    }

    fn tiny_moe(rng: &mut StdRng, device: &Device) -> SparseMoeBlock {
        let d = MoeDims {
            hidden_size: HIDDEN,
            num_experts: 6,
            moe_intermediate_size: 12,
            shared_expert_intermediate_size: 10,
        };
        let gate = Linear::new(rnd(&[d.num_experts, d.hidden_size], rng, device), None);
        let seg = Linear::new(rnd(&[1, d.hidden_size], rng, device), None);
        // Option J: pre-fused [2*inter, hidden] gate+up.
        let shared = SharedExpert::new(
            Linear::new(
                rnd(
                    &[2 * d.shared_expert_intermediate_size, d.hidden_size],
                    rng,
                    device,
                ),
                None,
            )
            .into(),
            Linear::new(
                rnd(
                    &[d.hidden_size, d.shared_expert_intermediate_size],
                    rng,
                    device,
                ),
                None,
            )
            .into(),
            d.shared_expert_intermediate_size,
        );
        let switch = SwitchMlp::new(
            rnd(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                rng,
                device,
            ),
            rnd(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                rng,
                device,
            ),
            rnd(
                &[d.num_experts, d.hidden_size, d.moe_intermediate_size],
                rng,
                device,
            ),
            d,
        )
        .unwrap();
        SparseMoeBlock::new(
            SparseMoeRuntime {
                dims: d,
                top_k: 3,
                norm_topk_prob: true,
            },
            gate.into(),
            seg.into(),
            shared,
            switch.into(),
        )
    }

    fn tiny_norm(device: &Device) -> RmsNorm {
        let w = Tensor::from_vec(vec![1f32; HIDDEN], (HIDDEN,), device).unwrap();
        RmsNorm::new(w, 1e-6)
    }

    /// Build a 40-layer tiny synthetic model following the same `is_linear` pattern as the
    /// real config (3 linear + 1 full × 10). Weight values are random; the goal is plumbing.
    fn build_tiny_model(device: &Device) -> Qwen3_5MoeTextModel {
        let mut r = StdRng::seed_from_u64(0xA11CE_BEEF);
        let embed = Embedding::new(rnd(&[VOCAB, HIDDEN], &mut r, device), HIDDEN);

        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        assert_eq!(cfg.text_config.num_hidden_layers, 40);

        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        for ty in cfg.text_config.layer_types.iter().copied() {
            let attention = match ty {
                LayerType::LinearAttention => {
                    AttentionBlock::Linear(tiny_linear_attn(&mut r, device))
                }
                LayerType::FullAttention => AttentionBlock::Full(tiny_self_attn(&mut r, device)),
            };
            layers.push(DecoderLayer::new(
                tiny_norm(device),
                attention,
                tiny_norm(device),
                tiny_moe(&mut r, device),
            ));
        }

        let lm_head: ProjLinear =
            candle_nn::Linear::new(rnd(&[VOCAB, HIDDEN], &mut r, device), None).into();
        Qwen3_5MoeTextModel::new(embed, layers, tiny_norm(device), lm_head)
    }

    fn is_finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }

    #[test]
    fn forty_layer_tiny_model_forward_returns_vocab_logits() {
        let device = Device::Cpu;
        let mut model = build_tiny_model(&device);
        assert_eq!(model.num_layers(), 40);

        // Input: [B=1, S=3] token IDs in vocab range.
        let input_ids = Tensor::from_vec(vec![0u32, 5, 12], (1, 3), &device).unwrap();
        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.dims(), &[1, 3, VOCAB]);
        assert!(is_finite(&logits), "forty-layer forward should be finite");
    }

    #[test]
    fn layer_dispatch_follows_three_one_pattern() {
        let device = Device::Cpu;
        let mut model = build_tiny_model(&device);
        let linear_indices: Vec<usize> = model
            .layers()
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.is_linear().then_some(i))
            .collect();
        let full_indices: Vec<usize> = model
            .layers()
            .iter()
            .enumerate()
            .filter_map(|(i, l)| (!l.is_linear()).then_some(i))
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert_eq!(linear_indices.len(), 30);
    }

    /// CB Phase 0 correctness: `forward_batch_decode_seqs_phase0` must match
    /// `forward_batch_decode_seqs` (v1) bit-equivalently — phase0 is just
    /// per-seq forward with batched final_norm/lm_head, no kernel-level
    /// changes.
    #[test]
    fn batch_decode_phase0_matches_v1() {
        let device = Device::Cpu;

        let build = || {
            let mut m = build_tiny_model(&device);
            m.enable_kv_cache(32);
            m
        };
        let prefill = |m: &mut Qwen3_5MoeTextModel, seq_id: u64, toks: &[u32]| {
            m.init_sequence(seq_id);
            m.set_current_seq_id(seq_id);
            let t = Tensor::from_slice(toks, (1, toks.len()), &device).unwrap();
            m.forward_with_offset(&t, 0).unwrap();
        };

        let mut model_v1 = build();
        let mut model_p0 = build();

        prefill(&mut model_v1, 1, &[1u32, 2, 3]);
        prefill(&mut model_v1, 2, &[4u32, 5, 6, 7]);
        prefill(&mut model_p0, 1, &[1u32, 2, 3]);
        prefill(&mut model_p0, 2, &[4u32, 5, 6, 7]);

        let seq_ids = [1u64, 2u64];
        let last_tokens = [8u32, 9u32];
        let positions = [3usize, 4usize];

        let out_v1 = model_v1
            .forward_batch_decode_seqs(&last_tokens, &seq_ids, &positions)
            .unwrap();
        let out_p0 = model_p0
            .forward_batch_decode_seqs_phase0(&last_tokens, &seq_ids, &positions)
            .unwrap();

        assert_eq!(out_v1.dims(), out_p0.dims());
        let diff = out_v1
            .sub(&out_p0)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "v1 vs phase0 max|Δ| = {diff:.2e} — phase0 must match v1 (no kernel change)"
        );
    }

    /// CB Phase 2 correctness: `forward_batch_decode_seqs_v2` must produce the same
    /// `[B, vocab]` logits as `forward_batch_decode_seqs` on identical model state.
    /// Two model instances are loaded from the same seed, prefilled identically, then
    /// a single decode step is compared — max|Δ| must be < 1e-4 (tiny f32 accumulation
    /// differences can arise from Tensor::cat reordering adds).
    #[test]
    fn batch_decode_v2_matches_v1() {
        let device = Device::Cpu;

        let build = || {
            let mut m = build_tiny_model(&device);
            m.enable_kv_cache(32);
            m
        };

        let prefill = |m: &mut Qwen3_5MoeTextModel, seq_id: u64, toks: &[u32]| {
            m.init_sequence(seq_id);
            m.set_current_seq_id(seq_id);
            let t = Tensor::from_slice(toks, (1, toks.len()), &device).unwrap();
            m.forward_with_offset(&t, 0).unwrap();
        };

        let mut model_v1 = build();
        let mut model_v2 = build();

        // Prefill two sequences on both models identically.
        prefill(&mut model_v1, 1, &[1u32, 2, 3]);
        prefill(&mut model_v1, 2, &[4u32, 5, 6, 7]);
        prefill(&mut model_v2, 1, &[1u32, 2, 3]);
        prefill(&mut model_v2, 2, &[4u32, 5, 6, 7]);

        let seq_ids = [1u64, 2u64];
        let last_tokens = [8u32, 9u32];
        let positions = [3usize, 4usize];

        let out_v1 = model_v1
            .forward_batch_decode_seqs(&last_tokens, &seq_ids, &positions)
            .unwrap();
        let out_v2 = model_v2
            .forward_batch_decode_seqs_v2(&last_tokens, &seq_ids, &positions)
            .unwrap();

        assert_eq!(out_v1.dims(), out_v2.dims(), "output shapes must match");
        let diff = out_v1
            .sub(&out_v2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "v1 vs v2 max|Δ| = {diff:.2e} — batched MoE must be numerically equivalent"
        );
    }

    /// Guard against the "embedding forwarded but stack skipped" regression: forward with
    /// random weights must NOT be bit-equal to `lm_head(final_norm(embed(ids)))` — that
    /// would mean the 40 layers did nothing.
    #[test]
    fn stack_contributes_to_output_not_just_embedding() {
        let device = Device::Cpu;
        let mut model = build_tiny_model(&device);
        let input_ids = Tensor::from_vec(vec![1u32, 7], (1, 2), &device).unwrap();

        let logits = model.forward(&input_ids).unwrap();
        let embed_only = model
            .lm_head
            .forward(
                &model
                    .final_norm
                    .forward(&model.embed_tokens.forward(&input_ids).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let diff = (&logits - &embed_only)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-4,
            "decoder stack should perturb the logits; got diff={diff}"
        );
    }
}
