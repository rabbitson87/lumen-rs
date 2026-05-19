//! TurboQuant Metal GPU Kernels — Zero-copy, batched command buffer architecture.

/// Metal type shim — re-exports the same Metal wrappers Candle uses
/// (`candle-metal-kernels::metal`, which is built on `objc2-metal`). This
/// is what every dependent (`lumen-model::qwen3_5_moe_native`,
/// `paged-attention`, etc.) links against, guaranteeing ABI compatibility
/// with Candle tensors so zero-copy bridges work.
pub mod metal;

pub mod affine3_gpu;
pub mod affine4_gpu;
#[cfg(feature = "model-integration")]
pub mod affine4_linear;
pub mod affine8_gpu;
#[cfg(feature = "model-integration")]
pub mod affine8_linear;
pub mod buffer;
pub mod device;
#[cfg(feature = "model-integration")]
pub mod flash_attn;
#[cfg(feature = "model-integration")]
pub mod gated_delta;
pub mod kernels;
#[cfg(feature = "mpsgraph")]
pub mod mpsgraph;
pub mod mxfp4;
pub mod mxfp4_gpu;
#[cfg(feature = "model-integration")]
pub mod mxfp4_linear;
pub mod pipeline;
#[cfg(feature = "model-integration")]
pub mod rms_norm;
pub mod sampling;
pub mod silu_mul;

use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use anyhow::Result;
use lumen_core::config::TurboQuantConfig;
use lumen_core::lloyd_max::LloydMaxCodebook;
use lumen_core::qjl::QJLProjector;
use lumen_core::rotation::RotationMatrix;

use buffer::{CompressedKVPool, KVPoolConfig};
use device::MetalContext;
use pipeline::ShaderPipelines;

/// GPU-accelerated TurboQuant compressor.
pub struct GpuCompressor {
    pub ctx: MetalContext,
    pub pipelines: ShaderPipelines,
    pub pool: CompressedKVPool,
    pub config: TurboQuantConfig,

    rotation_buf: crate::metal::Buffer,
    boundaries_buf: crate::metal::Buffer,
    centroids_buf: crate::metal::Buffer,
    qjl_matrix_buf: crate::metal::Buffer,
    rotation: RotationMatrix,

    // Pre-allocated scratch buffers (avoid per-call allocation)
    scratch_rotated: crate::metal::Buffer, // [max_heads * dim] f32
    scratch_codes: crate::metal::Buffer,   // [max_heads * dim] u8
    scratch_residuals: crate::metal::Buffer, // [max_heads * dim] f32
    scratch_tmp_packed: crate::metal::Buffer, // [max_heads * n_packed] u64
    scratch_tmp_scales: crate::metal::Buffer, // [max_heads] f32
    scratch_tmp_rn: crate::metal::Buffer,  // [max_heads] f32
    scratch_tmp_qjl: crate::metal::Buffer, // [max_heads * n_qjl_packed] u64
    scratch_scores: crate::metal::Buffer,  // [max_seq] f32
    scratch_output: crate::metal::Buffer,  // [max_heads * dim] f32
    scratch_qjl_proj: crate::metal::Buffer, // [qjl_m] f32 — precomputed per attention call
}

impl GpuCompressor {
    pub fn new(
        config: TurboQuantConfig,
        n_layers: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let pipelines = ShaderPipelines::new(&ctx.device)?;

        let rotation = RotationMatrix::random(config.head_dim, config.seed);
        let codebook = LloydMaxCodebook::compute(config.bits, config.lloyd_max_iter)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let qjl = QJLProjector::new(config.head_dim, config.qjl_m, config.seed.wrapping_add(1));

        let rotation_buf = ctx.buffer_with_data(&rotation.matrix);
        let boundaries_f32: Vec<f32> = codebook.boundaries.iter().map(|&x| x as f32).collect();
        let centroids_f32: Vec<f32> = codebook.centroids.iter().map(|&x| x as f32).collect();
        let boundaries_buf = ctx.buffer_with_data(&boundaries_f32);
        let centroids_buf = ctx.buffer_with_data(&centroids_f32);
        let qjl_matrix_buf = ctx.buffer_with_data(&qjl.proj_matrix);

        let pool_config = KVPoolConfig {
            max_seq_len,
            head_dim: config.head_dim,
            n_kv_heads,
            n_layers,
            bits: config.bits,
            qjl_m: config.qjl_m,
        };
        // Pre-allocate scratch buffers (sized for worst case: all heads at once)
        let dim = config.head_dim;
        let n_packed = pool_config.n_packed();
        let n_qjl_packed = pool_config.n_qjl_packed();

        let pool = CompressedKVPool::new(&ctx, pool_config);
        let max_heads = n_kv_heads.max(32); // support up to 32 Q heads

        // scratch sizing.
        // `store_kv` batches `total_vecs = n_kv_head × seq_len` into one
        // compress dispatch. Decode (seq_len=1) keeps it at `n_kv_head` (≤ 32),
        // but prefill (seq_len up to `max_seq_len`) can be `n_kv_head ×
        // max_seq_len`. The prior code sized scratch as `max_heads × dim`,
        // implicitly assuming `n_vecs ≤ max_heads`. For prefill that is OFF by
        // a factor of `max_seq_len` — the compress kernel writes far past the
        // end of these buffers, corrupting adjacent Metal memory (slot 0's
        // res_norm dumps as 1.2e18, slot 1+ scales all zero/NaN). bench_phase8
        // never caught this because it uses CPU compress; production prefill
        // never validated GPU compress correctness on the batched-vec path.
        // Size scratch for the worst case: `n_kv_head × max_seq_len` vectors.
        let max_vecs_per_call = n_kv_heads * max_seq_len;
        let max_vecs_for_alloc = max_vecs_per_call.max(max_heads);

        let scratch_rotated = ctx.buffer_for::<f32>(max_vecs_for_alloc * dim);
        let scratch_codes = ctx.buffer_for::<u8>(max_vecs_for_alloc * dim);
        let scratch_residuals = ctx.buffer_for::<f32>(max_vecs_for_alloc * dim);
        let scratch_tmp_packed = ctx.buffer_for::<u64>(max_vecs_for_alloc * n_packed);
        let scratch_tmp_scales = ctx.buffer_for::<f32>(max_vecs_for_alloc);
        let scratch_tmp_rn = ctx.buffer_for::<f32>(max_vecs_for_alloc);
        let scratch_tmp_qjl = ctx.buffer_for::<u64>(max_vecs_for_alloc * n_qjl_packed);
        let scratch_scores = ctx.buffer_for::<f32>(max_seq_len);
        let scratch_output = ctx.buffer_for::<f32>(max_heads * dim);
        let scratch_qjl_proj = ctx.buffer_for::<f32>(config.qjl_m);

        Ok(Self {
            ctx,
            pipelines,
            pool,
            config,
            rotation_buf,
            boundaries_buf,
            centroids_buf,
            qjl_matrix_buf,
            rotation,
            scratch_rotated,
            scratch_codes,
            scratch_residuals,
            scratch_tmp_packed,
            scratch_tmp_scales,
            scratch_tmp_rn,
            scratch_tmp_qjl,
            scratch_scores,
            scratch_output,
            scratch_qjl_proj,
        })
    }

    /// Compress KV vectors from CPU data (used by tests).
    pub fn compress_and_store(
        &mut self,
        layer: usize,
        head: usize,
        kv_vectors: &[f32],
        n_vecs: usize,
        is_key: bool,
    ) -> Result<()> {
        let dim = self.config.head_dim;
        if n_vecs > 0 && kv_vectors.len() / n_vecs != dim {
            return Ok(());
        }

        let n_packed = self.pool.config.n_packed();
        let n_qjl_packed = self.pool.config.n_qjl_packed();
        let n_levels = self.pool.config.n_levels();

        let kv_buf = self.ctx.buffer_with_data(kv_vectors);
        kernels::compress::compress_vectors(
            &self.ctx,
            &self.pipelines,
            &kv_buf,
            &self.rotation_buf,
            &self.boundaries_buf,
            &self.centroids_buf,
            &self.qjl_matrix_buf,
            &self.scratch_tmp_packed,
            &self.scratch_tmp_scales,
            &self.scratch_tmp_rn,
            &self.scratch_tmp_qjl,
            dim as u32,
            n_vecs as u32,
            self.config.bits,
            n_levels,
            n_packed as u32,
            self.config.qjl_m as u32,
            n_qjl_packed as u32,
        )?;
        self.blit_to_pool(layer, head, is_key, n_vecs);
        Ok(())
    }

    fn blit_to_pool(&mut self, layer: usize, head: usize, is_key: bool, n_vecs: usize) {
        let n_packed = self.pool.config.n_packed();
        let n_qjl_packed = self.pool.config.n_qjl_packed();
        let head_buf = self.pool.head_mut(layer, head);
        let (off, target) = if is_key {
            (head_buf.key_seq_len, &head_buf.key)
        } else {
            (head_buf.val_seq_len, &head_buf.value)
        };

        {
            let mut blit = crate::metal::process_commands()
                .blit_command_encoder()
                .expect("blit_command_encoder");
            blit.copy_from_buffer(
                &self.scratch_tmp_packed,
                0,
                &target.packed_codes,
                (off * n_packed * 8) as usize,
                (n_vecs * n_packed * 8) as usize,
            );
            blit.copy_from_buffer(
                &self.scratch_tmp_scales,
                0,
                &target.scales,
                (off * 4) as usize,
                (n_vecs * 4) as usize,
            );
            blit.copy_from_buffer(
                &self.scratch_tmp_rn,
                0,
                &target.res_norms,
                (off * 4) as usize,
                (n_vecs * 4) as usize,
            );
            blit.copy_from_buffer(
                &self.scratch_tmp_qjl,
                0,
                &target.qjl_bits,
                (off * n_qjl_packed * 8) as usize,
                (n_vecs * n_qjl_packed * 8) as usize,
            );
            // BlitCommandsGuard auto-end on drop
        }
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        let head_buf = self.pool.head_mut(layer, head);
        if is_key {
            head_buf.key_seq_len += n_vecs;
        } else {
            head_buf.val_seq_len += n_vecs;
        }
    }

    /// Compute attention scores (CPU path, tests).
    pub fn attention_scores(&self, layer: usize, head: usize, query: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.head_dim;
        if query.len() != dim {
            return Ok(vec![]);
        }
        let n_kv = self.pool.head(layer, head).key_seq_len;
        if n_kv == 0 {
            return Ok(vec![]);
        }

        let rq = self.rotation.apply(query);
        let rq_buf = self.ctx.buffer_with_data(&rq);
        let q_buf = self.ctx.buffer_with_data(query);
        let scores_buf = self.ctx.buffer_for::<f32>(n_kv);
        let hb = self.pool.head(layer, head);
        kernels::attention::compressed_attention_scores(
            &self.ctx,
            &self.pipelines,
            &rq_buf,
            &q_buf,
            &hb.key.packed_codes,
            &hb.key.scales,
            &self.centroids_buf,
            &hb.key.qjl_bits,
            &self.qjl_matrix_buf,
            &hb.key.res_norms,
            &scores_buf,
            dim as u32,
            n_kv as u32,
            self.config.bits,
            self.pool.config.n_packed() as u32,
            self.config.qjl_m as u32,
            self.pool.config.n_qjl_packed() as u32,
            self.pool.config.n_levels(),
        )?;
        Ok(self.ctx.read_buffer(&scores_buf, n_kv))
    }

    /// Value gather (CPU path, tests).
    pub fn value_gather(&self, layer: usize, head: usize, weights: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.head_dim;
        let n_kv = self.pool.head(layer, head).val_seq_len;
        assert_eq!(weights.len(), n_kv);
        let w_buf = self.ctx.buffer_with_data(weights);
        let out_buf = self.ctx.buffer_for::<f32>(dim);
        let hb = self.pool.head(layer, head);
        kernels::attention::compressed_value_gather(
            &self.ctx,
            &self.pipelines,
            &w_buf,
            &hb.value.packed_codes,
            &hb.value.scales,
            &self.centroids_buf,
            &self.rotation_buf,
            &out_buf,
            dim as u32,
            n_kv as u32,
            self.config.bits,
            self.pool.config.n_packed() as u32,
            self.pool.config.n_levels(),
        )?;
        Ok(self.ctx.read_buffer(&out_buf, dim))
    }

    pub fn clear_cache(&mut self) {
        self.pool.clear();
    }
}

// ── Candle integration: zero-copy GPU path ───────────────────────────────

#[cfg(feature = "model-integration")]
mod candle_integration {
    use super::*;
    use candle_core::{DType, Storage, Tensor};
    use candle_transformers::models::quantized_gemma4::CompressedKVBackend;

    fn set_u32(enc: &crate::metal::ComputeCommandEncoderRef, idx: u64, val: u32) {
        let bytes = val.to_ne_bytes();
        enc.set_bytes_directly(idx as usize, 4, bytes.as_ptr() as *const _);
    }

    /// Get Metal buffer reference and byte offset from a contiguous Candle tensor.
    /// Returns None if tensor is not on Metal.
    ///
    /// SAFETY: Candle uses a different version of the `metal` crate than us.
    /// We transmute the buffer reference — this is safe because crate::metal::Buffer
    /// is a transparent wrapper around objc::runtime::Object and the ABI is identical.
    fn get_metal_buffer(t: &Tensor) -> Option<(&crate::metal::Buffer, usize)> {
        let (storage_guard, layout) = t.storage_and_layout();
        match &*storage_guard {
            Storage::Metal(ms) => {
                let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
                let candle_buf = ms.buffer();
                // Transmute between metal crate versions (same underlying objc object)
                let buf_ptr = candle_buf as *const _ as *const crate::metal::Buffer;
                let buf_ref = unsafe { &*buf_ptr };
                Some((buf_ref, offset_bytes))
            }
            _ => None,
        }
    }

    impl CompressedKVBackend for GpuCompressor {
        fn store_kv(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> bool {
            let dims = k.dims();
            if dims.len() != 4 {
                return false;
            }
            let (n_kv_head, seq_len, head_dim) = (dims[1], dims[2], dims[3]);
            if head_dim != self.config.head_dim {
                return false;
            }

            let dim = head_dim;
            let n_packed = self.pool.config.n_packed();
            let n_qjl_packed = self.pool.config.n_qjl_packed();
            let n_levels = self.pool.config.n_levels();

            // Ensure F32 + contiguous for direct Metal buffer access
            let k_f32 = k.to_dtype(DType::F32).unwrap().contiguous().unwrap();
            let v_f32 = v.to_dtype(DType::F32).unwrap().contiguous().unwrap();

            // Get Metal buffers directly — ZERO CPU copy
            let (k_buf, k_base) = match get_metal_buffer(&k_f32) {
                Some(b) => b,
                None => {
                    // CPU fallback
                    return self.store_kv_cpu(layer, &k_f32, &v_f32, n_kv_head, seq_len, dim);
                }
            };
            let (v_buf, v_base) = get_metal_buffer(&v_f32).unwrap();

            // cross-queue hazard fix.
            // k/v come from upstream Candle ops (RoPE / qknorm / qkv split),
            // queued on Candle's command queue. Our compress kernel runs on
            // the TurboQuant queue. Without a sync, our queue may read the
            // K/V buffers before Candle has finished writing them → garbage
            // codes / scales / qjl-bits in the pool → garbage scores at
            // decode → compressed_attention returns near-zero output.
            // Same hazard as in `compressed_attention`. Flush Candle queue
            // before our compress dispatch.
            let _ = k.device().synchronize();

            // Layout: [1, n_kv_head, seq_len, head_dim] contiguous
            // Head h data starts at base + h * seq_len * dim * 4 bytes
            let head_stride_bytes = (seq_len * dim * 4) as u64;

            // Batch compress: all KV heads in 2 calls (K + V) instead of 8 (per-head)
            // Layout [1, n_kv_head, seq_len, dim] is contiguous → n_vecs = n_kv_head * seq_len
            let total_vecs = (n_kv_head * seq_len) as u32;

            // Compress ALL K heads in one call (4 kernel dispatches instead of 16)
            kernels::compress::encode_compress(
                &self.ctx,
                &self.pipelines,
                k_buf,
                k_base,
                &self.rotation_buf,
                &self.boundaries_buf,
                &self.centroids_buf,
                &self.qjl_matrix_buf,
                &self.scratch_tmp_packed,
                0,
                &self.scratch_tmp_scales,
                0,
                &self.scratch_tmp_rn,
                0,
                &self.scratch_tmp_qjl,
                0,
                &self.scratch_rotated,
                &self.scratch_codes,
                &self.scratch_residuals,
                dim as u32,
                total_vecs,
                self.config.bits,
                n_levels,
                n_packed as u32,
                self.config.qjl_m as u32,
                n_qjl_packed as u32,
            )
            .unwrap();

            // Blit K results to per-head pool locations
            for h in 0..n_kv_head {
                let hb = self.pool.head(layer, h);
                let k_off = hb.key_seq_len;
                let src_vec_off = h * seq_len; // offset within batched output
                let mut blit = crate::metal::process_commands()
                    .blit_command_encoder()
                    .expect("blit");
                blit.copy_from_buffer(
                    &self.scratch_tmp_packed,
                    (src_vec_off * n_packed * 8) as usize,
                    &hb.key.packed_codes,
                    (k_off * n_packed * 8) as usize,
                    (seq_len * n_packed * 8) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_scales,
                    (src_vec_off * 4) as usize,
                    &hb.key.scales,
                    (k_off * 4) as usize,
                    (seq_len * 4) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_rn,
                    (src_vec_off * 4) as usize,
                    &hb.key.res_norms,
                    (k_off * 4) as usize,
                    (seq_len * 4) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_qjl,
                    (src_vec_off * n_qjl_packed * 8) as usize,
                    &hb.key.qjl_bits,
                    (k_off * n_qjl_packed * 8) as usize,
                    (seq_len * n_qjl_packed * 8) as usize,
                );
            }

            // Compress ALL V heads in one call
            kernels::compress::encode_compress(
                &self.ctx,
                &self.pipelines,
                v_buf,
                v_base,
                &self.rotation_buf,
                &self.boundaries_buf,
                &self.centroids_buf,
                &self.qjl_matrix_buf,
                &self.scratch_tmp_packed,
                0,
                &self.scratch_tmp_scales,
                0,
                &self.scratch_tmp_rn,
                0,
                &self.scratch_tmp_qjl,
                0,
                &self.scratch_rotated,
                &self.scratch_codes,
                &self.scratch_residuals,
                dim as u32,
                total_vecs,
                self.config.bits,
                n_levels,
                n_packed as u32,
                self.config.qjl_m as u32,
                n_qjl_packed as u32,
            )
            .unwrap();

            // Blit V results to per-head pool locations
            for h in 0..n_kv_head {
                let hb = self.pool.head(layer, h);
                let v_off = hb.val_seq_len;
                let src_vec_off = h * seq_len;
                let mut blit = crate::metal::process_commands()
                    .blit_command_encoder()
                    .expect("blit");
                blit.copy_from_buffer(
                    &self.scratch_tmp_packed,
                    (src_vec_off * n_packed * 8) as usize,
                    &hb.value.packed_codes,
                    (v_off * n_packed * 8) as usize,
                    (seq_len * n_packed * 8) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_scales,
                    (src_vec_off * 4) as usize,
                    &hb.value.scales,
                    (v_off * 4) as usize,
                    (seq_len * 4) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_rn,
                    (src_vec_off * 4) as usize,
                    &hb.value.res_norms,
                    (v_off * 4) as usize,
                    (seq_len * 4) as usize,
                );
                blit.copy_from_buffer(
                    &self.scratch_tmp_qjl,
                    (src_vec_off * n_qjl_packed * 8) as usize,
                    &hb.value.qjl_bits,
                    (v_off * n_qjl_packed * 8) as usize,
                    (seq_len * n_qjl_packed * 8) as usize,
                );
            }

            // ONE sync for ALL heads
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");

            // Update seq_lens
            for h in 0..n_kv_head {
                let hb = self.pool.head_mut(layer, h);
                hb.key_seq_len += seq_len;
                hb.val_seq_len += seq_len;
            }
            true
        }

        fn compressed_attention(
            &mut self,
            layer: usize,
            q: &Tensor,
            n_head: usize,
            n_kv_head: usize,
            head_dim: usize,
        ) -> Option<Tensor> {
            if head_dim != self.config.head_dim {
                return None;
            }
            let n_kv = self.pool.head(layer, 0).key_seq_len;
            if n_kv == 0 {
                return None;
            }

            let dim = head_dim;
            let gqa_ratio = n_head / n_kv_head;
            let n_packed = self.pool.config.n_packed() as u32;
            let n_qjl_packed = self.pool.config.n_qjl_packed() as u32;
            let n_levels = self.pool.config.n_levels();

            // dump pool scales/res_norms to
            // localise outlier source. Reads from pool buffer for layer 0 head 0
            // first n_kv vecs. Activated by LUMEN_TQ_DEBUG_DUMP=1.
            if std::env::var("LUMEN_TQ_DEBUG_DUMP")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                let hb = self.pool.head(layer, 0);
                let scales_vec: Vec<f32> = self.ctx.read_buffer(&hb.key.scales, n_kv);
                let rn_vec: Vec<f32> = self.ctx.read_buffer(&hb.key.res_norms, n_kv);
                let s_min = scales_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let s_max = scales_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let s_nan = scales_vec.iter().filter(|v| !v.is_finite()).count();
                let r_min = rn_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let r_max = rn_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let r_nan = rn_vec.iter().filter(|v| !v.is_finite()).count();
                eprintln!(
                    "[TQ-DUMP] L{layer:02} h0 K n_kv={n_kv} scales: min={s_min:.3e} max={s_max:.3e} nan={s_nan} | res_norms: min={r_min:.3e} max={r_max:.3e} nan={r_nan} | scales[0..min(5,n_kv)]={:?}",
                    &scales_vec[..scales_vec.len().min(5)]
                );
            }

            // GQA-correct compressed attention.
            // Legacy code (default) reads ONE Q vector per GQA group (the first)
            // and fan-outs the same softmax·V to all gqa_ratio output slots —
            // that is MQA, not GQA. Real model Q heads in a group are
            // differentiated; discarding gqa_ratio-1 of them collapses softmax
            // and produces token-0 spam (B4.1 production validation, anti-pattern #19).
            // LUMEN_TQ_GQA_FIX=1 enables a per-Q-head loop that reuses K/V from
            // kv_h=qh/gqa_ratio but computes a distinct softmax per Q head.
            // ~gqa_ratio× more dispatches; trades speed for correctness.
            let gqa_fix = std::env::var("LUMEN_TQ_GQA_FIX")
                .map(|v| v == "1")
                .unwrap_or(false);

            // missing 1/sqrt(d_k) attention scale.
            // The score kernel (`tq_compressed_attention_scores_v6`) computes the
            // raw dot product Q·K (stage1 + QJL correction) but never multiplies
            // by `1/sqrt(d_k)`. `bench_phase8_e2e` applied this scale on host
            // before host-side softmax, but production sends raw scores straight
            // into `tq_softmax_parallel` → softmax saturates (sqrt(256)=16
            // magnitude → exp(16) ≈ 8.9M) → near-Dirac attention → output ≈
            // V[argmax] → o_proj → constant logit pattern → token-0 spam.
            // Fix: pre-scale Q by 1/sqrt(d_k) so all downstream linear ops
            // (rotation + stage1 codebook recon + QJL projection) inherit the
            // factor without kernel-level changes. `LUMEN_TQ_SCALE_FIX=1`
            // gates the fix so legacy MQA path remains bit-for-bit identical
            // when both flags are unset.
            let scale_fix = std::env::var("LUMEN_TQ_SCALE_FIX")
                .map(|v| v == "1")
                .unwrap_or(false);

            // Get Q Metal buffer — zero copy
            let q_raw = q.to_dtype(DType::F32).ok()?;
            let q_f32 = if scale_fix {
                let inv_sqrt = 1.0f64 / (dim as f64).sqrt();
                (q_raw * inv_sqrt).ok()?.contiguous().ok()?
            } else {
                q_raw.contiguous().ok()?
            };
            let (q_buf, q_base) = get_metal_buffer(&q_f32)?;

            // Pre-allocate output as Candle Tensor on same device (zero copy output)
            let output_tensor = Tensor::zeros((1, n_head, 1, dim), DType::F32, q.device())
                .ok()?
                .contiguous()
                .ok()?;
            // cross-queue hazard fix.
            // `Tensor::zeros(...)` queues a Metal zero-init op on the Candle
            // command queue. Our `compressed_attention` writes via the
            // TurboQuant Metal queue. Without an explicit sync, the two
            // queues race — Candle's zero-init may execute AFTER our writes,
            // silently overwriting them. Symptom: `compressed_attention`
            // returns exact zeros (cos vs ref = 0.0000 exact, max|Δ| = max|ref|).
            // Same hazard documented in `playbook_kv_cache_attention.md`
            // (cross-queue zero-copy). Same applies to the scaled q_f32 tensor
            // when LUMEN_TQ_SCALE_FIX is on (mul-by-scalar via Candle).
            // Fix: flush Candle queue before our compute kernels start.
            q.device().synchronize().ok()?;
            let (out_buf, out_base) = get_metal_buffer(&output_tensor)?;

            let rq_slots = if gqa_fix { n_head } else { n_kv_head };
            let rq_buf = self.ctx.buffer_for::<f32>(rq_slots * dim);

            // Per-Q-head scratch for the GQA-fix path; legacy path skips these.
            let qjl_proj_per_q_buf = if gqa_fix {
                Some(self.ctx.buffer_for::<f32>(n_head * self.config.qjl_m))
            } else {
                None
            };
            let scores_per_q_buf = if gqa_fix {
                Some(self.ctx.buffer_for::<f32>(n_head * n_kv))
            } else {
                None
            };

            // === ONE command buffer for EVERYTHING ===

            // Rotate queries on GPU
            // Note: compute encoders auto-end via Drop; no explicit end_encoding().
            for slot in 0..rq_slots {
                let qh = if gqa_fix { slot } else { slot * gqa_ratio };
                let enc = crate::metal::process_commands()
                    .command_encoder()
                    .expect("ce");
                let p = self.pipelines.get("tq_rotate_query").ok()?;
                enc.set_compute_pipeline_state(p);
                enc.set_buffer(0, Some(q_buf), q_base + (qh * dim * 4) as usize);
                enc.set_buffer(1, Some(&self.rotation_buf), (0) as usize);
                enc.set_buffer(2, Some(&rq_buf), (slot * dim * 4) as usize);
                set_u32(enc.as_ref(), 3, dim as u32);
                set_u32(enc.as_ref(), 4, 1);
                enc.dispatch_threads(
                    crate::mtl_size!(dim as usize, 1, 1),
                    crate::mtl_size!(dim.min(256) as u64, 1, 1),
                );
            }

            // GQA-correct path: per-Q-head loop.
            if gqa_fix {
                let qjl_proj_per_q = qjl_proj_per_q_buf.as_ref().unwrap();
                let scores_per_q = scores_per_q_buf.as_ref().unwrap();
                for qh in 0..n_head {
                    let kv_h = qh / gqa_ratio;
                    let hb = self.pool.head(layer, kv_h);

                    // QJL project Q[qh]
                    {
                        let enc = crate::metal::process_commands()
                            .command_encoder()
                            .expect("ce");
                        let p = self.pipelines.get("tq_qjl_project_query").ok()?;
                        enc.set_compute_pipeline_state(p);
                        enc.set_buffer(0, Some(q_buf), q_base + (qh * dim * 4) as usize);
                        enc.set_buffer(1, Some(&self.qjl_matrix_buf), 0);
                        enc.set_buffer(
                            2,
                            Some(qjl_proj_per_q),
                            (qh * self.config.qjl_m * 4) as usize,
                        );
                        set_u32(enc.as_ref(), 3, dim as u32);
                        set_u32(enc.as_ref(), 4, self.config.qjl_m as u32);
                        let max_tg = p.max_total_threads_per_threadgroup() as u64;
                        let qjl_m = self.config.qjl_m as u64;
                        let tg = qjl_m.min(max_tg);
                        enc.dispatch_threads(
                            crate::mtl_size!(qjl_m, 1, 1),
                            crate::mtl_size!(tg, 1, 1),
                        );
                    }

                    // Scores: rq[qh] · K_comp[kv_h]
                    {
                        let enc = crate::metal::process_commands()
                            .command_encoder()
                            .expect("ce");
                        let use_v6 = dim % 4 == 0;
                        let kernel_name = if use_v6 {
                            "tq_compressed_attention_scores_v6"
                        } else {
                            "tq_compressed_attention_scores_v2"
                        };
                        let p = self.pipelines.get(kernel_name).ok()?;
                        enc.set_compute_pipeline_state(p);
                        enc.set_buffer(0, Some(&rq_buf), (qh * dim * 4) as usize);
                        enc.set_buffer(
                            1,
                            Some(qjl_proj_per_q),
                            (qh * self.config.qjl_m * 4) as usize,
                        );
                        enc.set_buffer(2, Some(&hb.key.packed_codes), 0);
                        enc.set_buffer(3, Some(&hb.key.scales), 0);
                        enc.set_buffer(4, Some(&self.centroids_buf), 0);
                        enc.set_buffer(5, Some(&hb.key.qjl_bits), 0);
                        enc.set_buffer(6, Some(&hb.key.res_norms), 0);
                        enc.set_buffer(7, Some(scores_per_q), (qh * n_kv * 4) as usize);
                        set_u32(enc.as_ref(), 8, dim as u32);
                        set_u32(enc.as_ref(), 9, n_kv as u32);
                        set_u32(enc.as_ref(), 10, self.config.bits);
                        set_u32(enc.as_ref(), 11, n_packed);
                        set_u32(enc.as_ref(), 12, self.config.qjl_m as u32);
                        set_u32(enc.as_ref(), 13, n_qjl_packed);
                        set_u32(enc.as_ref(), 14, n_levels);
                        if use_v6 {
                            const SIMD_SIZE: u64 = 32;
                            const KV_PER_TG: u64 = 8;
                            let tg_size = SIMD_SIZE * KV_PER_TG;
                            let num_tgs = ((n_kv as u64) + KV_PER_TG - 1) / KV_PER_TG;
                            enc.dispatch_thread_groups(
                                crate::mtl_size!(num_tgs, 1, 1),
                                crate::mtl_size!(tg_size, 1, 1),
                            );
                        } else {
                            let max_tg = p.max_total_threads_per_threadgroup() as u64;
                            let tg = (n_kv as u64).min(max_tg);
                            enc.dispatch_threads(
                                crate::mtl_size!(n_kv, 1, 1),
                                crate::mtl_size!(tg, 1, 1),
                            );
                        }
                    }

                    // Softmax scores[qh]
                    {
                        let enc = crate::metal::process_commands()
                            .command_encoder()
                            .expect("ce");
                        let p = self.pipelines.get("tq_softmax_parallel").ok()?;
                        enc.set_compute_pipeline_state(p);
                        enc.set_buffer(0, Some(scores_per_q), (qh * n_kv * 4) as usize);
                        set_u32(enc.as_ref(), 1, n_kv as u32);
                        let max_tg = p.max_total_threads_per_threadgroup() as u64;
                        let tg = (n_kv as u64).next_power_of_two().min(max_tg);
                        enc.dispatch_thread_groups(
                            crate::mtl_size!(1, 1, 1),
                            crate::mtl_size!(tg, 1, 1),
                        );
                    }

                    // Single-output value gather → out[qh]
                    {
                        let enc = crate::metal::process_commands()
                            .command_encoder()
                            .expect("ce");
                        let p = self.pipelines.get("tq_compressed_value_gather").ok()?;
                        enc.set_compute_pipeline_state(p);
                        enc.set_buffer(0, Some(scores_per_q), (qh * n_kv * 4) as usize);
                        enc.set_buffer(1, Some(&hb.value.packed_codes), 0);
                        enc.set_buffer(2, Some(&hb.value.scales), 0);
                        enc.set_buffer(3, Some(&self.centroids_buf), 0);
                        enc.set_buffer(4, Some(&self.rotation_buf), 0);
                        enc.set_buffer(5, Some(out_buf), out_base + (qh * dim * 4) as usize);
                        set_u32(enc.as_ref(), 6, dim as u32);
                        set_u32(enc.as_ref(), 7, n_kv as u32);
                        set_u32(enc.as_ref(), 8, self.config.bits);
                        set_u32(enc.as_ref(), 9, n_packed);
                        set_u32(enc.as_ref(), 10, n_levels);
                        enc.dispatch_thread_groups(
                            crate::mtl_size!(1, 1, 1),
                            crate::mtl_size!(dim as usize, 1, 1),
                        );
                    }
                }

                crate::metal::process_commands()
                    .flush_and_wait()
                    .expect("flush");
                return Some(output_tensor);
            }

            // Legacy MQA-broadcast path.
            // Per KV-head: scores + softmax + value gather
            for kv_h in 0..n_kv_head {
                let hb = self.pool.head(layer, kv_h);
                let first_qh = kv_h * gqa_ratio;

                // Precompute QJL projection of query for this KV head:
                //   scratch_qjl_proj[j] = qjl_matrix[j] · query
                // Replaces the per-kv_idx O(qjl_m * dim) recomputation in v1 score kernel.
                {
                    let enc = crate::metal::process_commands()
                        .command_encoder()
                        .expect("ce");
                    let p = self.pipelines.get("tq_qjl_project_query").ok()?;
                    enc.set_compute_pipeline_state(p);
                    enc.set_buffer(0, Some(q_buf), q_base + (first_qh * dim * 4) as usize);
                    enc.set_buffer(1, Some(&self.qjl_matrix_buf), 0);
                    enc.set_buffer(2, Some(&self.scratch_qjl_proj), 0);
                    set_u32(enc.as_ref(), 3, dim as u32);
                    set_u32(enc.as_ref(), 4, self.config.qjl_m as u32);
                    let max_tg = p.max_total_threads_per_threadgroup() as u64;
                    let qjl_m = self.config.qjl_m as u64;
                    let tg = qjl_m.min(max_tg);
                    enc.dispatch_threads(crate::mtl_size!(qjl_m, 1, 1), crate::mtl_size!(tg, 1, 1));
                }

                // Compressed attention scores via SIMD-group cooperative reduction (v6).
                // 1 simd-group (32 threads) per kv_idx, 8 kv_idx per threadgroup.
                // Multi-threadgroup grid correctly handles arbitrary n_kv.
                // Fallback to v2 when dim % 4 != 0 (v6 also requires the float4-style
                // contiguous load implicitly via simd-cooperative dim stride).
                {
                    let enc = crate::metal::process_commands()
                        .command_encoder()
                        .expect("ce");
                    let use_v6 = dim % 4 == 0;
                    let kernel_name = if use_v6 {
                        "tq_compressed_attention_scores_v6"
                    } else {
                        "tq_compressed_attention_scores_v2"
                    };
                    let p = self.pipelines.get(kernel_name).ok()?;
                    enc.set_compute_pipeline_state(p);
                    enc.set_buffer(0, Some(&rq_buf), (kv_h * dim * 4) as usize);
                    enc.set_buffer(1, Some(&self.scratch_qjl_proj), 0);
                    enc.set_buffer(2, Some(&hb.key.packed_codes), (0) as usize);
                    enc.set_buffer(3, Some(&hb.key.scales), (0) as usize);
                    enc.set_buffer(4, Some(&self.centroids_buf), (0) as usize);
                    enc.set_buffer(5, Some(&hb.key.qjl_bits), (0) as usize);
                    enc.set_buffer(6, Some(&hb.key.res_norms), (0) as usize);
                    enc.set_buffer(7, Some(&self.scratch_scores), (0) as usize);
                    set_u32(enc.as_ref(), 8, dim as u32);
                    set_u32(enc.as_ref(), 9, n_kv as u32);
                    set_u32(enc.as_ref(), 10, self.config.bits);
                    set_u32(enc.as_ref(), 11, n_packed);
                    set_u32(enc.as_ref(), 12, self.config.qjl_m as u32);
                    set_u32(enc.as_ref(), 13, n_qjl_packed);
                    set_u32(enc.as_ref(), 14, n_levels);
                    if use_v6 {
                        const SIMD_SIZE: u64 = 32;
                        const KV_PER_TG: u64 = 8;
                        let tg_size = SIMD_SIZE * KV_PER_TG;
                        let num_tgs = ((n_kv as u64) + KV_PER_TG - 1) / KV_PER_TG;
                        enc.dispatch_thread_groups(
                            crate::mtl_size!(num_tgs, 1, 1),
                            crate::mtl_size!(tg_size, 1, 1),
                        );
                    } else {
                        let max_tg = p.max_total_threads_per_threadgroup() as u64;
                        let tg = (n_kv as u64).min(max_tg);
                        enc.dispatch_threads(
                            crate::mtl_size!(n_kv, 1, 1),
                            crate::mtl_size!(tg, 1, 1),
                        );
                    }
                }

                // Softmax over scores. tq_softmax_parallel handles arbitrary n_kv
                // via single-threadgroup three-pass reduction (max → exp+sum → norm).
                // Fixes the silent-corruption bug of fused single-threadgroup softmax
                // when n_kv > max_threads_per_threadgroup.
                {
                    let enc = crate::metal::process_commands()
                        .command_encoder()
                        .expect("ce");
                    let p = self.pipelines.get("tq_softmax_parallel").ok()?;
                    enc.set_compute_pipeline_state(p);
                    enc.set_buffer(0, Some(&self.scratch_scores), 0);
                    set_u32(enc.as_ref(), 1, n_kv as u32);
                    let max_tg = p.max_total_threads_per_threadgroup() as u64;
                    let tg = (n_kv as u64).next_power_of_two().min(max_tg);
                    enc.dispatch_thread_groups(
                        crate::mtl_size!(1, 1, 1),
                        crate::mtl_size!(tg, 1, 1),
                    );
                }

                // Value gather with GQA fan-out: single compute, write to
                // gqa_ratio consecutive Q-head output slots in one dispatch.
                // All Q heads in this KV group share identical softmax weights
                // and V cache → identical compute → fan out result.
                {
                    let first_qh = kv_h * gqa_ratio;
                    let enc = crate::metal::process_commands()
                        .command_encoder()
                        .expect("ce");
                    let p = self
                        .pipelines
                        .get("tq_compressed_value_gather_multi")
                        .ok()?;
                    enc.set_compute_pipeline_state(p);
                    enc.set_buffer(0, Some(&self.scratch_scores), 0);
                    enc.set_buffer(1, Some(&hb.value.packed_codes), 0);
                    enc.set_buffer(2, Some(&hb.value.scales), 0);
                    enc.set_buffer(3, Some(&self.centroids_buf), 0);
                    enc.set_buffer(4, Some(&self.rotation_buf), 0);
                    enc.set_buffer(5, Some(out_buf), out_base + (first_qh * dim * 4) as usize);
                    set_u32(enc.as_ref(), 6, dim as u32);
                    set_u32(enc.as_ref(), 7, n_kv as u32);
                    set_u32(enc.as_ref(), 8, self.config.bits);
                    set_u32(enc.as_ref(), 9, n_packed);
                    set_u32(enc.as_ref(), 10, n_levels);
                    set_u32(enc.as_ref(), 11, gqa_ratio as u32);
                    enc.dispatch_thread_groups(
                        crate::mtl_size!(1, 1, 1),
                        crate::mtl_size!(dim as usize, 1, 1),
                    );
                }
            }

            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");

            // output_tensor already has the data on GPU — zero copy return
            Some(output_tensor)
        }

        fn seq_len(&self, layer: usize) -> usize {
            self.pool.head(layer, 0).seq_len()
        }

        fn clear(&mut self) {
            self.clear_cache();
        }

        fn truncate(&mut self, layer: usize, n_keep: usize) {
            self.pool.truncate(layer, n_keep);
        }
    }

    impl GpuCompressor {
        /// CPU fallback for store_kv when tensor is not on Metal.
        fn store_kv_cpu(
            &mut self,
            layer: usize,
            k: &Tensor,
            v: &Tensor,
            n_kv_head: usize,
            seq_len: usize,
            dim: usize,
        ) -> bool {
            for h in 0..n_kv_head {
                let kh = k
                    .narrow(1, h, 1)
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                let kh_flat: Vec<f32> = kh.into_iter().flatten().collect();
                let vh = v
                    .narrow(1, h, 1)
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                let vh_flat: Vec<f32> = vh.into_iter().flatten().collect();
                self.compress_and_store(layer, h, &kh_flat, seq_len, true)
                    .unwrap();
                self.compress_and_store(layer, h, &vh_flat, seq_len, false)
                    .unwrap();
            }
            true
        }
    }
}
