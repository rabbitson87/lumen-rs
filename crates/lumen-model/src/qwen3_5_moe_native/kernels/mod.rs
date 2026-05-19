//! Native kernels (Phase A.2+): embedding lookup, RMSNorm, etc.
//!
//! The shader source lives in [`shader.metal`] and is compiled once at
//! [`KernelLib::new`] time. Pipelines are cached on the lib and dispatched
//! against `NativeContext`'s queue so every op shares one cmd-buffer lineage.

use anyhow::{Result, anyhow};
use lumen_metal::metal::{
    BatchedEncoderExt, CommandBufferExt, CompileOptions, ComputeCommandEncoderRef,
    ComputeEncoderCompat, ComputePipelineState, Library, MTLLanguageVersion, MTLSize,
};

use super::context::NativeContext;
use super::tensor::{NativeDType, NativeTensor};

const SHADER_SRC: &str = include_str!("shader.metal");

pub struct KernelLib {
    #[allow(dead_code)]
    library: Library,
    embedding_lookup_f32: ComputePipelineState,
    rms_norm_f32: ComputePipelineState,
    rope_partial_f32: ComputePipelineState,
    attention_causal_f32: ComputePipelineState,
    transpose_blhd_f32: ComputePipelineState,
    ssm_step_f32: ComputePipelineState,
    rms_norm_weightless_f32: ComputePipelineState,
    softplus_f32: ComputePipelineState,
    silu_f32: ComputePipelineState,
    sigmoid_f32: ComputePipelineState,
    broadcast_add_per_head_f32: ComputePipelineState,
    mul_broadcast_per_head_f32: ComputePipelineState,
    neg_exp_f32: ComputePipelineState,
    compute_g_full_f32: ComputePipelineState,
    repeat_heads_blhd_f32: ComputePipelineState,
    affine_scalar_f32: ComputePipelineState,
    depthwise_conv1d_silu_f32: ComputePipelineState,
    silu_mul_f32: ComputePipelineState,
    sigmoid_mul_f32: ComputePipelineState,
    // Workstream B Phase 6 — bf16 SSM subgraph variants.
    rms_norm_weightless_bf16: ComputePipelineState,
    affine_scalar_bf16: ComputePipelineState,
    repeat_heads_blhd_bf16: ComputePipelineState,
    ssm_step_bf16: ComputePipelineState,
    cast_f32_to_bf16: ComputePipelineState,
    cast_bf16_to_f32: ComputePipelineState,
}

impl KernelLib {
    pub fn new(ctx: &NativeContext) -> Result<Self> {
        let options = lumen_metal::metal::new_compile_options();
        // bumps from 3.0 → 3.1 for `bfloat` type support. The 3.0
        // f32 kernels still compile under 3.1 (forward-compatible).
        options.set_language_version(MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow!("native kernels compile error: {e}"))?;

        let make_pso = |fn_name: &str| -> Result<ComputePipelineState> {
            let func = library
                .get_function(fn_name, None)
                .map_err(|e| anyhow!("kernel `{fn_name}` not found: {e}"))?;
            ctx.device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow!("pipeline `{fn_name}` failed: {e}"))
        };

        let embedding_lookup_f32 = make_pso("embedding_lookup_f32")?;
        let rms_norm_f32 = make_pso("rms_norm_f32")?;
        let rope_partial_f32 = make_pso("rope_partial_f32")?;
        let attention_causal_f32 = make_pso("attention_causal_f32")?;
        let transpose_blhd_f32 = make_pso("transpose_blhd_f32")?;
        let ssm_step_f32 = make_pso("ssm_step_f32")?;
        let rms_norm_weightless_f32 = make_pso("rms_norm_weightless_f32")?;
        let softplus_f32 = make_pso("softplus_f32")?;
        let silu_f32 = make_pso("silu_f32")?;
        let sigmoid_f32 = make_pso("sigmoid_f32")?;
        let broadcast_add_per_head_f32 = make_pso("broadcast_add_per_head_f32")?;
        let mul_broadcast_per_head_f32 = make_pso("mul_broadcast_per_head_f32")?;
        let neg_exp_f32 = make_pso("neg_exp_f32")?;
        let compute_g_full_f32 = make_pso("compute_g_full_f32")?;
        let repeat_heads_blhd_f32 = make_pso("repeat_heads_blhd_f32")?;
        let affine_scalar_f32 = make_pso("affine_scalar_f32")?;
        let depthwise_conv1d_silu_f32 = make_pso("depthwise_conv1d_silu_f32")?;
        let silu_mul_f32 = make_pso("silu_mul_f32")?;
        let sigmoid_mul_f32 = make_pso("sigmoid_mul_f32")?;
        let rms_norm_weightless_bf16 = make_pso("rms_norm_weightless_bf16")?;
        let affine_scalar_bf16 = make_pso("affine_scalar_bf16")?;
        let repeat_heads_blhd_bf16 = make_pso("repeat_heads_blhd_bf16")?;
        let ssm_step_bf16 = make_pso("ssm_step_bf16")?;
        let cast_f32_to_bf16 = make_pso("cast_f32_to_bf16")?;
        let cast_bf16_to_f32 = make_pso("cast_bf16_to_f32")?;

        Ok(Self {
            library,
            embedding_lookup_f32,
            rms_norm_f32,
            rope_partial_f32,
            attention_causal_f32,
            transpose_blhd_f32,
            ssm_step_f32,
            rms_norm_weightless_f32,
            softplus_f32,
            silu_f32,
            sigmoid_f32,
            broadcast_add_per_head_f32,
            mul_broadcast_per_head_f32,
            neg_exp_f32,
            compute_g_full_f32,
            repeat_heads_blhd_f32,
            affine_scalar_f32,
            depthwise_conv1d_silu_f32,
            silu_mul_f32,
            sigmoid_mul_f32,
            rms_norm_weightless_bf16,
            affine_scalar_bf16,
            repeat_heads_blhd_bf16,
            ssm_step_bf16,
            cast_f32_to_bf16,
            cast_bf16_to_f32,
        })
    }

    /// `out[t, h] = embed[token_ids[t], h]`.
    ///
    /// Shapes:
    ///   - `token_ids`: `[seq_len]` U32
    ///   - `embed`:     `[vocab, hidden]` F32
    ///   - `out`:       `[seq_len, hidden]` F32 (pre-allocated)
    pub fn embedding_lookup(
        &self,
        ctx: &NativeContext,
        token_ids: &NativeTensor,
        embed: &NativeTensor,
        out: &NativeTensor,
    ) -> Result<()> {
        if token_ids.dtype() != NativeDType::U32 {
            return Err(anyhow!(
                "embedding_lookup token_ids dtype {:?} != U32",
                token_ids.dtype()
            ));
        }
        if embed.dtype() != NativeDType::F32 || out.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "embedding_lookup f32 only (got embed={:?}, out={:?})",
                embed.dtype(),
                out.dtype()
            ));
        }
        if token_ids.rank() != 1 {
            return Err(anyhow!(
                "embedding_lookup token_ids must be rank 1, got shape {:?}",
                token_ids.shape()
            ));
        }
        if embed.rank() != 2 {
            return Err(anyhow!(
                "embedding_lookup embed must be rank 2, got shape {:?}",
                embed.shape()
            ));
        }
        let seq_len = token_ids.shape()[0];
        let hidden = embed.shape()[1];
        if out.shape() != [seq_len, hidden] {
            return Err(anyhow!(
                "embedding_lookup out shape {:?} != [{}, {}]",
                out.shape(),
                seq_len,
                hidden
            ));
        }
        if seq_len == 0 || hidden == 0 {
            return Ok(());
        }

        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:embedding_lookup");
        enc.set_compute_pipeline_state(&self.embedding_lookup_f32);
        enc.set_buffer(0, Some(token_ids.buffer()), token_ids.offset_bytes());
        enc.set_buffer(1, Some(embed.buffer()), embed.offset_bytes());
        enc.set_buffer(2, Some(out.buffer()), out.offset_bytes());
        let hidden_u32 = hidden as u32;
        enc.set_bytes_directly(3, 4, &hidden_u32 as *const _ as *const _);

        let max_threads = self
            .embedding_lookup_f32
            .max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = MTLSize {
            width: hidden as usize,
            height: seq_len as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// `y[r, :] = (x[r, :] * rsqrt(mean(x[r, :]^2) + eps)) * gamma`.
    ///
    /// Shapes:
    ///   - `x`:     `[rows, hidden]` F32
    ///   - `gamma`: `[hidden]` F32
    ///   - `y`:     `[rows, hidden]` F32 (pre-allocated)
    pub fn rms_norm(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        gamma: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        // Validate BEFORE creating the encoder so an early `Err` doesn't drop
        // an encoder mid-encoding (which Metal asserts on).
        let prep = self.validate_rms_norm(x, gamma, y)?;
        if prep.is_none() {
            return Ok(());
        }
        let (rows, hidden, tg_size) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:rms_norm");
        self.dispatch_rms_norm(&enc, x, gamma, eps, y, rows, hidden, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only variant: dispatches `rms_norm` into an existing compute
    /// encoder without committing. Caller is responsible for the encoder
    /// (`end_encoding`) and command buffer (`commit` / `wait_until_completed`)
    /// lifecycle. Use this when fusing multiple kernels into one command
    /// buffer to amortize the per-commit overhead (~50µs each on M3 Max).
    pub fn encode_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        gamma: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_rms_norm(x, gamma, y)?;
        if let Some((rows, hidden, tg_size)) = prep {
            self.dispatch_rms_norm(&enc, x, gamma, eps, y, rows, hidden, tg_size);
        }
        Ok(())
    }

    /// Validation helper. Returns `Some((rows, hidden, tg_size))` when the
    /// dispatch should proceed, `None` when the inputs are zero-sized and the
    /// kernel should be skipped, and `Err` on shape/dtype mismatch.
    fn validate_rms_norm(
        &self,
        x: &NativeTensor,
        gamma: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<Option<(usize, usize, usize)>> {
        if x.dtype() != NativeDType::F32
            || gamma.dtype() != NativeDType::F32
            || y.dtype() != NativeDType::F32
        {
            return Err(anyhow!("rms_norm f32 only"));
        }
        if x.rank() != 2 {
            return Err(anyhow!(
                "rms_norm x must be rank 2 [rows, hidden], got {:?}",
                x.shape()
            ));
        }
        if gamma.rank() != 1 {
            return Err(anyhow!(
                "rms_norm gamma must be rank 1, got {:?}",
                gamma.shape()
            ));
        }
        let rows = x.shape()[0];
        let hidden = x.shape()[1];
        if gamma.shape() != [hidden] {
            return Err(anyhow!(
                "rms_norm gamma shape {:?} != [{}]",
                gamma.shape(),
                hidden
            ));
        }
        if y.shape() != [rows, hidden] {
            return Err(anyhow!(
                "rms_norm y shape {:?} != [{}, {}]",
                y.shape(),
                rows,
                hidden
            ));
        }
        if rows == 0 || hidden == 0 {
            return Ok(None);
        }

        let max_threads = self.rms_norm_f32.max_total_threads_per_threadgroup();
        let want = (hidden as usize).min(max_threads).min(256);
        let mut tg_size: usize = 1;
        while tg_size * 2 <= want {
            tg_size *= 2;
        }
        if tg_size == 0 {
            tg_size = 1;
        }
        Ok(Some((rows, hidden, tg_size)))
    }

    fn dispatch_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        gamma: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
        rows: usize,
        hidden: usize,
        tg_size: usize,
    ) {
        enc.set_compute_pipeline_state(&self.rms_norm_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(gamma.buffer()), gamma.offset_bytes());
        enc.set_buffer(2, Some(y.buffer()), y.offset_bytes());
        let hidden_u32 = hidden as u32;
        enc.set_bytes_directly(3, 4, &hidden_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &eps as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, tg_size * 4);

        let grid_groups = MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
    }

    /// Apply GPT-NeoX split-form rotary embedding to the first `2*half_d` dims of
    /// every head. Components past `rotary_dim` pass through unchanged.
    ///
    /// Shapes:
    ///   - `x`, `y`:   `[batch, seq, heads, head_dim]` F32 (must NOT alias)
    ///   - `cos`,`sin`: `[seq, half_d]` F32
    ///
    /// `half_d * 2 == rotary_dim`. Caller chooses `rotary_dim` from
    /// `partial_rotary_factor × head_dim` (snapped even).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        cos: &NativeTensor,
        sin: &NativeTensor,
        y: &NativeTensor,
        half_d: usize,
    ) -> Result<()> {
        for (name, t) in [("x", x), ("cos", cos), ("sin", sin), ("y", y)] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!("rope_partial: {name} dtype {:?} != F32", t.dtype()));
            }
        }
        if x.rank() != 4 {
            return Err(anyhow!(
                "rope_partial x must be rank 4 [B, S, H, D], got {:?}",
                x.shape()
            ));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "rope_partial x/y shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let heads = x.shape()[2];
        let head_dim = x.shape()[3];
        if cos.shape() != [seq, half_d] || sin.shape() != [seq, half_d] {
            return Err(anyhow!(
                "rope_partial cos/sin must be [seq={}, half_d={}], got cos {:?} sin {:?}",
                seq,
                half_d,
                cos.shape(),
                sin.shape()
            ));
        }
        if 2 * half_d > head_dim {
            return Err(anyhow!(
                "rope_partial 2*half_d={} exceeds head_dim={}",
                2 * half_d,
                head_dim
            ));
        }
        // SAFETY contract: caller must not alias x and y. Each thread reads
        // from a paired offset that another thread writes — in-place would
        // create a data race. We don't enforce this at runtime because metal
        // Buffer doesn't expose pointer equality directly without a `ForeignType`
        // import. Invariant violations show up as parity test failures.
        if batch == 0 || seq == 0 || heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:rope_partial");
        self.encode_rope_partial_inner(&enc, x, cos, sin, y, batch, seq, heads, head_dim, half_d);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only `rope_partial`. See [`Self::encode_rms_norm`] for the
    /// fusion rationale. Returns the same validation errors as
    /// [`Self::rope_partial`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_partial(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        cos: &NativeTensor,
        sin: &NativeTensor,
        y: &NativeTensor,
        half_d: usize,
    ) -> Result<()> {
        for (name, t) in [("x", x), ("cos", cos), ("sin", sin), ("y", y)] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!("rope_partial: {name} dtype {:?} != F32", t.dtype()));
            }
        }
        if x.rank() != 4 {
            return Err(anyhow!(
                "rope_partial x must be rank 4 [B, S, H, D], got {:?}",
                x.shape()
            ));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "rope_partial x/y shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let heads = x.shape()[2];
        let head_dim = x.shape()[3];
        if cos.shape() != [seq, half_d] || sin.shape() != [seq, half_d] {
            return Err(anyhow!(
                "rope_partial cos/sin must be [seq={}, half_d={}], got cos {:?} sin {:?}",
                seq,
                half_d,
                cos.shape(),
                sin.shape()
            ));
        }
        if 2 * half_d > head_dim {
            return Err(anyhow!(
                "rope_partial 2*half_d={} exceeds head_dim={}",
                2 * half_d,
                head_dim
            ));
        }
        if batch == 0 || seq == 0 || heads == 0 || head_dim == 0 {
            return Ok(());
        }
        self.encode_rope_partial_inner(&enc, x, cos, sin, y, batch, seq, heads, head_dim, half_d);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rope_partial_inner(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        cos: &NativeTensor,
        sin: &NativeTensor,
        y: &NativeTensor,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        half_d: usize,
    ) {
        enc.set_compute_pipeline_state(&self.rope_partial_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(cos.buffer()), cos.offset_bytes());
        enc.set_buffer(2, Some(sin.buffer()), sin.offset_bytes());
        enc.set_buffer(3, Some(y.buffer()), y.offset_bytes());
        let seq_u32 = seq as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let half_u32 = half_d as u32;
        enc.set_bytes_directly(4, 4, &seq_u32 as *const _ as *const _);
        enc.set_bytes_directly(5, 4, &heads_u32 as *const _ as *const _);
        enc.set_bytes_directly(6, 4, &head_dim_u32 as *const _ as *const _);
        enc.set_bytes_directly(7, 4, &half_u32 as *const _ as *const _);

        let max_threads = self.rope_partial_f32.max_total_threads_per_threadgroup();
        let tx = max_threads.min(head_dim as usize).max(1);
        let grid = MTLSize {
            width: head_dim as usize,
            height: heads as usize,
            depth: (batch * seq) as usize,
        };
        let tg = MTLSize {
            width: tx,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

impl KernelLib {
    /// Causal-masked attention: `out = softmax(Q · K^T / sqrt(D) + mask) · V`.
    ///
    /// GQA-aware: `q_heads` may exceed `kv_heads` as long as `q_heads % kv_heads
    /// == 0`. Each Q head `q_h` shares K/V head `q_h / (q_heads / kv_heads)`,
    /// matching the `repeat_interleave` semantics used by Candle's `qwen3_5_moe`.
    ///
    /// Shapes:
    ///   - `q`:   `[B, q_heads, L_q,  D]`
    ///   - `k, v`:`[B, kv_heads, L_kv, D]`
    ///   - `out`: `[B, q_heads, L_q,  D]` (pre-allocated)
    ///
    /// `pos_offset`: position of `q[..., 0, :]` in the absolute sequence — lets
    /// decode reuse the kernel by skipping the prompt's already-cached keys.
    /// Causal mask = `score(q_idx, k) = -inf if k > pos_offset + q_idx`.
    ///
    /// Single-tile baseline. `l_kv * 4` must fit threadgroup memory (32 KB
    /// on Apple GPU). Long-context tiling lands in a later step.
    #[allow(clippy::too_many_arguments)]
    /// Wraps the `attention_causal_f32` kernel with the legacy always-causal contract
    /// (matches earlier callers / parity tests). New code should prefer
    /// [`Self::attention`] which exposes the `apply_causal` flag explicitly.
    pub fn attention_causal(
        &self,
        ctx: &NativeContext,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        out: &NativeTensor,
        pos_offset: usize,
    ) -> Result<()> {
        self.attention(ctx, q, k, v, out, pos_offset, true)
    }

    /// GQA scaled-dot-product attention with selectable causal masking. When
    /// `apply_causal` is `false`, every query attends to every key — matching
    /// Candle's production prefill (`mask=None` in `model.rs::forward_with_offset`).
    ///
    /// K/V are assumed contiguous over the L axis (`kv_layout_stride = l_kv`).
    /// Use [`Self::attention_with_kv_stride`] when K/V live in a pre-allocated
    /// cache buffer whose physical L capacity exceeds the active sequence length.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        ctx: &NativeContext,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        out: &NativeTensor,
        pos_offset: usize,
        apply_causal: bool,
    ) -> Result<()> {
        let l_kv = k.shape()[2];
        self.attention_with_kv_stride(ctx, q, k, v, out, pos_offset, apply_causal, l_kv, l_kv)
    }

    /// Variant of [`Self::attention`] that decouples the active K/V sequence
    /// length (`l_kv`) from the physical layout stride (`kv_layout_stride`).
    ///
    /// When K/V come from a `NativeKvCache` of shape `[B, kv, max_seq, D]` but
    /// only the first `current_seq_len` rows are populated, callers pass:
    ///
    ///   - `l_kv = current_seq_len` — what the kernel attends to,
    ///   - `kv_layout_stride = max_seq` — the inter-head physical stride.
    ///
    /// The K/V tensors must have shape `[B, kv_heads, kv_layout_stride, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_with_kv_stride(
        &self,
        ctx: &NativeContext,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        out: &NativeTensor,
        pos_offset: usize,
        apply_causal: bool,
        l_kv: usize,
        kv_layout_stride: usize,
    ) -> Result<()> {
        for (name, t) in [("q", q), ("k", k), ("v", v), ("out", out)] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!(
                    "attention_causal: {name} dtype {:?} != F32",
                    t.dtype()
                ));
            }
            if t.rank() != 4 {
                return Err(anyhow!(
                    "attention_causal: {name} must be rank 4 [B,H,L,D], got {:?}",
                    t.shape()
                ));
            }
        }
        let (b, q_heads, l_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let kv_heads = k.shape()[1];
        if k.shape() != [b, kv_heads, kv_layout_stride, d]
            || v.shape() != [b, kv_heads, kv_layout_stride, d]
        {
            return Err(anyhow!(
                "attention_causal: K/V shape mismatch (expected [{b},{kv_heads},{kv_layout_stride},{d}], \
                 got K={:?}, V={:?})",
                k.shape(),
                v.shape()
            ));
        }
        if l_kv > kv_layout_stride {
            return Err(anyhow!(
                "attention_causal: l_kv ({l_kv}) > kv_layout_stride ({kv_layout_stride})"
            ));
        }
        if out.shape() != [b, q_heads, l_q, d] {
            return Err(anyhow!(
                "attention_causal: out shape {:?} != [{b},{q_heads},{l_q},{d}]",
                out.shape()
            ));
        }
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(anyhow!(
                "attention_causal: q_heads ({q_heads}) must be a positive multiple of kv_heads ({kv_heads})"
            ));
        }
        if b == 0 || l_q == 0 || l_kv == 0 || d == 0 {
            return Ok(());
        }

        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:tq_flash_attn");
        self.encode_attention_inner(
            &enc,
            q,
            k,
            v,
            out,
            b,
            q_heads,
            kv_heads,
            l_q,
            l_kv,
            d,
            pos_offset,
            apply_causal,
            kv_layout_stride,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only `attention_with_kv_stride`. Validation behavior matches the
    /// committing variant.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_with_kv_stride(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        out: &NativeTensor,
        pos_offset: usize,
        apply_causal: bool,
        l_kv: usize,
        kv_layout_stride: usize,
    ) -> Result<()> {
        for (name, t) in [("q", q), ("k", k), ("v", v), ("out", out)] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!(
                    "attention_causal: {name} dtype {:?} != F32",
                    t.dtype()
                ));
            }
            if t.rank() != 4 {
                return Err(anyhow!(
                    "attention_causal: {name} must be rank 4 [B,H,L,D], got {:?}",
                    t.shape()
                ));
            }
        }
        let (b, q_heads, l_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let kv_heads = k.shape()[1];
        if k.shape() != [b, kv_heads, kv_layout_stride, d]
            || v.shape() != [b, kv_heads, kv_layout_stride, d]
        {
            return Err(anyhow!(
                "attention_causal: K/V shape mismatch (expected [{b},{kv_heads},{kv_layout_stride},{d}], \
                 got K={:?}, V={:?})",
                k.shape(),
                v.shape()
            ));
        }
        if l_kv > kv_layout_stride {
            return Err(anyhow!(
                "attention_causal: l_kv ({l_kv}) > kv_layout_stride ({kv_layout_stride})"
            ));
        }
        if out.shape() != [b, q_heads, l_q, d] {
            return Err(anyhow!(
                "attention_causal: out shape {:?} != [{b},{q_heads},{l_q},{d}]",
                out.shape()
            ));
        }
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(anyhow!(
                "attention_causal: q_heads ({q_heads}) must be a positive multiple of kv_heads ({kv_heads})"
            ));
        }
        if b == 0 || l_q == 0 || l_kv == 0 || d == 0 {
            return Ok(());
        }
        self.encode_attention_inner(
            enc,
            q,
            k,
            v,
            out,
            b,
            q_heads,
            kv_heads,
            l_q,
            l_kv,
            d,
            pos_offset,
            apply_causal,
            kv_layout_stride,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_inner(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        out: &NativeTensor,
        b: usize,
        q_heads: usize,
        kv_heads: usize,
        l_q: usize,
        l_kv: usize,
        d: usize,
        pos_offset: usize,
        apply_causal: bool,
        kv_layout_stride: usize,
    ) {
        let scale = 1.0_f32 / (d as f32).sqrt();
        let max_threads = self
            .attention_causal_f32
            .max_total_threads_per_threadgroup();
        let tg_w = (d as usize).min(max_threads).min(256).max(1);

        enc.set_compute_pipeline_state(&self.attention_causal_f32);
        enc.set_buffer(0, Some(q.buffer()), q.offset_bytes());
        enc.set_buffer(1, Some(k.buffer()), k.offset_bytes());
        enc.set_buffer(2, Some(v.buffer()), v.offset_bytes());
        enc.set_buffer(3, Some(out.buffer()), out.offset_bytes());
        let q_h_u32 = q_heads as u32;
        let kv_h_u32 = kv_heads as u32;
        let l_q_u32 = l_q as u32;
        let l_kv_u32 = l_kv as u32;
        let d_u32 = d as u32;
        let pos_u32 = pos_offset as u32;
        enc.set_bytes_directly(4, 4, &q_h_u32 as *const _ as *const _);
        enc.set_bytes_directly(5, 4, &kv_h_u32 as *const _ as *const _);
        enc.set_bytes_directly(6, 4, &l_q_u32 as *const _ as *const _);
        enc.set_bytes_directly(7, 4, &l_kv_u32 as *const _ as *const _);
        enc.set_bytes_directly(8, 4, &d_u32 as *const _ as *const _);
        enc.set_bytes_directly(9, 4, &pos_u32 as *const _ as *const _);
        enc.set_bytes_directly(10, 4, &scale as *const _ as *const _);
        let tg_size_u32 = tg_w as u32;
        enc.set_bytes_directly(11, 4, &tg_size_u32 as *const _ as *const _);
        let apply_causal_u32: u32 = if apply_causal { 1 } else { 0 };
        enc.set_bytes_directly(12, 4, &apply_causal_u32 as *const _ as *const _);
        let kv_stride_u32 = kv_layout_stride as u32;
        enc.set_bytes_directly(13, 4, &kv_stride_u32 as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, (l_kv * 4) as usize);

        let grid_groups = MTLSize {
            width: l_q as usize,
            height: q_heads as usize,
            depth: b as usize,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
    }

    /// Transpose `[B, L, H, D]` → `[B, H, L, D]` (`direction = 0`) or the
    /// inverse (`direction = 1`). `x` and `y` must have shapes consistent with
    /// the requested direction.
    pub fn transpose_blhd(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        y: &NativeTensor,
        direction: u32,
    ) -> Result<()> {
        if x.dtype() != NativeDType::F32 || y.dtype() != NativeDType::F32 {
            return Err(anyhow!("transpose_blhd: F32 only"));
        }
        if x.rank() != 4 || y.rank() != 4 {
            return Err(anyhow!(
                "transpose_blhd: rank 4 required (got x={:?}, y={:?})",
                x.shape(),
                y.shape()
            ));
        }
        let (b, l, h, d) = if direction == 0 {
            (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3])
        } else {
            // BHLD → BLHD: swap meanings.
            (x.shape()[0], x.shape()[2], x.shape()[1], x.shape()[3])
        };
        let expected_y = if direction == 0 {
            [b, h, l, d]
        } else {
            [b, l, h, d]
        };
        if y.shape() != expected_y {
            return Err(anyhow!(
                "transpose_blhd: y shape {:?} does not match expected {:?} (dir={})",
                y.shape(),
                expected_y,
                direction
            ));
        }
        if b == 0 || l == 0 || h == 0 || d == 0 {
            return Ok(());
        }

        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:transpose_blhd");
        self.encode_transpose_blhd_inner(&enc, x, y, b, l, h, d, direction);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only `transpose_blhd`. Validation matches the committing variant.
    pub fn encode_transpose_blhd(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
        direction: u32,
    ) -> Result<()> {
        if x.dtype() != NativeDType::F32 || y.dtype() != NativeDType::F32 {
            return Err(anyhow!("transpose_blhd: F32 only"));
        }
        if x.rank() != 4 || y.rank() != 4 {
            return Err(anyhow!(
                "transpose_blhd: rank 4 required (got x={:?}, y={:?})",
                x.shape(),
                y.shape()
            ));
        }
        let (b, l, h, d) = if direction == 0 {
            (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3])
        } else {
            (x.shape()[0], x.shape()[2], x.shape()[1], x.shape()[3])
        };
        let expected_y = if direction == 0 {
            [b, h, l, d]
        } else {
            [b, l, h, d]
        };
        if y.shape() != expected_y {
            return Err(anyhow!(
                "transpose_blhd: y shape {:?} does not match expected {:?} (dir={})",
                y.shape(),
                expected_y,
                direction
            ));
        }
        if b == 0 || l == 0 || h == 0 || d == 0 {
            return Ok(());
        }
        self.encode_transpose_blhd_inner(&enc, x, y, b, l, h, d, direction);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_transpose_blhd_inner(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
        b: usize,
        l: usize,
        h: usize,
        d: usize,
        direction: u32,
    ) {
        enc.set_compute_pipeline_state(&self.transpose_blhd_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let l_u32 = l as u32;
        let h_u32 = h as u32;
        let d_u32 = d as u32;
        enc.set_bytes_directly(2, 4, &l_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &h_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &d_u32 as *const _ as *const _);
        enc.set_bytes_directly(5, 4, &direction as *const _ as *const _);

        let max_threads = self.transpose_blhd_f32.max_total_threads_per_threadgroup();
        let tx = max_threads.min(d as usize).max(1);
        let grid = MTLSize {
            width: d as usize,
            height: (l * h) as usize,
            depth: b as usize,
        };
        let tg = MTLSize {
            width: tx,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

impl KernelLib {
    /// One step of the GatedDeltaNet SSM update (Phase A.4).
    ///
    /// Mutates `state` in place and writes a new output `y_t`. Caller drives
    /// the seq-loop by invoking this kernel `seq_len` times with the same
    /// `state` buffer and per-step `(q_t, k_t, v_t, beta_t, g_t)`. The
    /// multi-step fused variant lands in a follow-up step.
    ///
    /// Shapes:
    ///   - `state` (in/out): `[B, Hv, Dv, Dk]` F32
    ///   - `q`, `k`:        `[B, Hv, Dk]`     F32
    ///   - `v`:             `[B, Hv, Dv]`     F32
    ///   - `beta`, `g`:     `[B, Hv]`         F32
    ///   - `y` (out):       `[B, Hv, Dv]`     F32
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_step(
        &self,
        ctx: &NativeContext,
        state: &NativeTensor,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        beta: &NativeTensor,
        g: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_ssm_step(state, q, k, v, beta, g, y)?;
        if prep.is_none() {
            return Ok(());
        }
        let (b, hv, dv, dk, tg_size) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:tq_gated_delta_step");
        self.dispatch_ssm_step(&enc, state, q, k, v, beta, g, y, b, hv, dv, dk, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only `ssm_step`. Same fusion rationale as
    /// [`Self::encode_rms_norm`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ssm_step(
        &self,
        enc: &ComputeCommandEncoderRef,
        state: &NativeTensor,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        beta: &NativeTensor,
        g: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((b, hv, dv, dk, tg_size)) =
            self.validate_ssm_step(state, q, k, v, beta, g, y)?
        {
            self.dispatch_ssm_step(&enc, state, q, k, v, beta, g, y, b, hv, dv, dk, tg_size);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_ssm_step(
        &self,
        state: &NativeTensor,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        beta: &NativeTensor,
        g: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<Option<(usize, usize, usize, usize, usize)>> {
        for (name, t) in [
            ("state", state),
            ("q", q),
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("g", g),
            ("y", y),
        ] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!("ssm_step: {name} dtype {:?} != F32", t.dtype()));
            }
        }
        if state.rank() != 4 {
            return Err(anyhow!(
                "ssm_step state must be rank 4 [B,Hv,Dv,Dk], got {:?}",
                state.shape()
            ));
        }
        let (b, hv, dv, dk) = (
            state.shape()[0],
            state.shape()[1],
            state.shape()[2],
            state.shape()[3],
        );
        if q.shape() != [b, hv, dk] || k.shape() != [b, hv, dk] {
            return Err(anyhow!(
                "ssm_step q/k must be [{b},{hv},{dk}], got q={:?} k={:?}",
                q.shape(),
                k.shape()
            ));
        }
        if v.shape() != [b, hv, dv] || y.shape() != [b, hv, dv] {
            return Err(anyhow!(
                "ssm_step v/y must be [{b},{hv},{dv}], got v={:?} y={:?}",
                v.shape(),
                y.shape()
            ));
        }
        if beta.shape() != [b, hv] || g.shape() != [b, hv] {
            return Err(anyhow!(
                "ssm_step beta/g must be [{b},{hv}], got beta={:?} g={:?}",
                beta.shape(),
                g.shape()
            ));
        }
        if b == 0 || hv == 0 || dv == 0 || dk == 0 {
            return Ok(None);
        }

        // Threadgroup width: power-of-2 ≤ min(Dk, max_threads, 256).
        let max_threads = self.ssm_step_f32.max_total_threads_per_threadgroup();
        let want = (dk as usize).min(max_threads).min(256);
        let mut tg_size: usize = 1;
        while tg_size * 2 <= want {
            tg_size *= 2;
        }
        if tg_size == 0 {
            tg_size = 1;
        }
        Ok(Some((b, hv, dv, dk, tg_size)))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_ssm_step(
        &self,
        enc: &ComputeCommandEncoderRef,
        state: &NativeTensor,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        beta: &NativeTensor,
        g: &NativeTensor,
        y: &NativeTensor,
        b: usize,
        hv: usize,
        dv: usize,
        dk: usize,
        tg_size: usize,
    ) {
        enc.set_compute_pipeline_state(&self.ssm_step_f32);
        enc.set_buffer(0, Some(state.buffer()), state.offset_bytes());
        enc.set_buffer(1, Some(q.buffer()), q.offset_bytes());
        enc.set_buffer(2, Some(k.buffer()), k.offset_bytes());
        enc.set_buffer(3, Some(v.buffer()), v.offset_bytes());
        enc.set_buffer(4, Some(beta.buffer()), beta.offset_bytes());
        enc.set_buffer(5, Some(g.buffer()), g.offset_bytes());
        enc.set_buffer(6, Some(y.buffer()), y.offset_bytes());
        let hv_u32 = hv as u32;
        let dv_u32 = dv as u32;
        let dk_u32 = dk as u32;
        let tg_u32 = tg_size as u32;
        enc.set_bytes_directly(7, 4, &hv_u32 as *const _ as *const _);
        enc.set_bytes_directly(8, 4, &dv_u32 as *const _ as *const _);
        enc.set_bytes_directly(9, 4, &dk_u32 as *const _ as *const _);
        enc.set_bytes_directly(10, 4, &tg_u32 as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, tg_size * 4);

        let grid_groups = MTLSize {
            width: dv as usize,
            height: hv as usize,
            depth: b as usize,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
        let _ = b;
        let _ = dk;
    }
}

impl KernelLib {
    /// Weightless RMSNorm on the last axis: `y = x / sqrt(mean(x^2) + eps)`.
    /// No `gamma` weight — used by `linear_attn`'s SSM Q/K normalization
    /// (`mx.fast.rms_norm(x, None, 1e-6)` in the upstream MLX reference).
    pub fn rms_norm_weightless(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_rms_norm_weightless(x, y)?;
        if prep.is_none() {
            return Ok(());
        }
        let (rows, hidden, tg_size) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:rms_norm_weightless");
        self.dispatch_rms_norm_weightless(&enc, x, eps, y, rows, hidden, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only `rms_norm_weightless`. Same fusion rationale as
    /// [`Self::encode_rms_norm`].
    pub fn encode_rms_norm_weightless(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((rows, hidden, tg_size)) = self.validate_rms_norm_weightless(x, y)? {
            self.dispatch_rms_norm_weightless(&enc, x, eps, y, rows, hidden, tg_size);
        }
        Ok(())
    }

    fn validate_rms_norm_weightless(
        &self,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<Option<(usize, usize, usize)>> {
        if x.dtype() != NativeDType::F32 || y.dtype() != NativeDType::F32 {
            return Err(anyhow!("rms_norm_weightless f32 only"));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "rms_norm_weightless shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        if x.rank() < 1 {
            return Err(anyhow!("rms_norm_weightless requires rank >= 1"));
        }
        let hidden = *x.shape().last().unwrap();
        let rows: usize = x.shape().iter().rev().skip(1).product();
        if rows == 0 || hidden == 0 {
            return Ok(None);
        }

        let max_threads = self
            .rms_norm_weightless_f32
            .max_total_threads_per_threadgroup();
        let want = (hidden as usize).min(max_threads).min(256);
        let mut tg_size: usize = 1;
        while tg_size * 2 <= want {
            tg_size *= 2;
        }
        if tg_size == 0 {
            tg_size = 1;
        }
        Ok(Some((rows, hidden, tg_size)))
    }

    fn dispatch_rms_norm_weightless(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
        rows: usize,
        hidden: usize,
        tg_size: usize,
    ) {
        enc.set_compute_pipeline_state(&self.rms_norm_weightless_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let hidden_u32 = hidden as u32;
        enc.set_bytes_directly(2, 4, &hidden_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &eps as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, tg_size * 4);

        let grid_groups = MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
    }

    fn dispatch_elementwise(
        &self,
        ctx: &NativeContext,
        pso: &ComputePipelineState,
        x: &NativeTensor,
        y: &NativeTensor,
        op_name: &str,
    ) -> Result<()> {
        let prep = Self::validate_elementwise(x, y, op_name)?;
        if prep.is_none() {
            return Ok(());
        }
        let (n, tg_w) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label(&format!("lumen:elementwise_{op_name}"));
        Self::dispatch_elementwise_inner(&enc, pso, x, y, n, tg_w);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn validate_elementwise(
        x: &NativeTensor,
        y: &NativeTensor,
        op_name: &str,
    ) -> Result<Option<(usize, usize)>> {
        if x.dtype() != NativeDType::F32 || y.dtype() != NativeDType::F32 {
            return Err(anyhow!("{op_name} f32 only"));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "{op_name} shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let n = x.numel();
        if n == 0 {
            return Ok(None);
        }
        // tg_w sizing depends on the pipeline's max_total_threads_per_threadgroup;
        // resolve against the actual pso in `encode_elementwise`. Cap at 256 here.
        let tg_w = (n as usize).min(256).max(1);
        Ok(Some((n, tg_w)))
    }

    fn dispatch_elementwise_inner(
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        x: &NativeTensor,
        y: &NativeTensor,
        n: usize,
        tg_w_hint: usize,
    ) {
        let max_threads = pso.max_total_threads_per_threadgroup();
        let tg_w = tg_w_hint.min(max_threads).max(1);
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(2, 4, &n_u32 as *const _ as *const _);

        let grid = MTLSize {
            width: n as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }

    fn encode_elementwise(
        &self,
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        x: &NativeTensor,
        y: &NativeTensor,
        op_name: &str,
    ) -> Result<()> {
        if let Some((n, tg_w)) = Self::validate_elementwise(x, y, op_name)? {
            Self::dispatch_elementwise_inner(&enc, pso, x, y, n, tg_w);
        }
        Ok(())
    }

    /// Element-wise softplus: `y = ln(1 + exp(x))` (numerically stable).
    pub fn softplus(&self, ctx: &NativeContext, x: &NativeTensor, y: &NativeTensor) -> Result<()> {
        self.dispatch_elementwise(ctx, &self.softplus_f32, x, y, "softplus")
    }

    /// Encode-only [`Self::softplus`].
    pub fn encode_softplus(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_elementwise(enc, &self.softplus_f32, x, y, "softplus")
    }

    /// Element-wise SiLU: `y = x * sigmoid(x)`.
    pub fn silu(&self, ctx: &NativeContext, x: &NativeTensor, y: &NativeTensor) -> Result<()> {
        self.dispatch_elementwise(ctx, &self.silu_f32, x, y, "silu")
    }

    /// Encode-only [`Self::silu`].
    pub fn encode_silu(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_elementwise(enc, &self.silu_f32, x, y, "silu")
    }

    /// Element-wise sigmoid: `y = 1 / (1 + exp(-x))`.
    pub fn sigmoid(&self, ctx: &NativeContext, x: &NativeTensor, y: &NativeTensor) -> Result<()> {
        self.dispatch_elementwise(ctx, &self.sigmoid_f32, x, y, "sigmoid")
    }

    /// Encode-only [`Self::sigmoid`].
    pub fn encode_sigmoid(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_elementwise(enc, &self.sigmoid_f32, x, y, "sigmoid")
    }

    /// Element-wise `y = exp(-x)`. Last step of the GatedDeltaNet decay chain.
    pub fn neg_exp(&self, ctx: &NativeContext, x: &NativeTensor, y: &NativeTensor) -> Result<()> {
        self.dispatch_elementwise(ctx, &self.neg_exp_f32, x, y, "neg_exp")
    }

    /// Encode-only [`Self::neg_exp`].
    pub fn encode_neg_exp(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_elementwise(enc, &self.neg_exp_f32, x, y, "neg_exp")
    }

    /// fused `compute_g + beta` kernel (5 → 1 dispatch).
    ///
    /// Replaces the chain
    ///   beta = sigmoid(b)
    ///   a_dt = a + dt_bias[h]
    ///   g    = exp(-softplus(a_dt) * exp_a_log[h])
    /// with a single element-wise kernel. Mirrors MLX's
    /// `@partial(mx.compile) compute_g` (gated_delta.py).
    ///
    /// Shape contract:
    ///   `b`, `a`, `beta_out`, `g_out`: same f32 shape `[B, S, Hv]` (or any flat
    ///       layout with last axis = Hv).
    ///   `dt_bias`, `exp_a_log`: f32 `[Hv]`.
    pub fn encode_compute_g_full(
        &self,
        enc: &ComputeCommandEncoderRef,
        b: &NativeTensor,
        a: &NativeTensor,
        dt_bias: &NativeTensor,
        exp_a_log: &NativeTensor,
        beta_out: &NativeTensor,
        g_out: &NativeTensor,
    ) -> Result<()> {
        for (name, t) in [
            ("b", b),
            ("a", a),
            ("dt_bias", dt_bias),
            ("exp_a_log", exp_a_log),
            ("beta_out", beta_out),
            ("g_out", g_out),
        ] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!(
                    "compute_g_full: {name} dtype {:?} != F32",
                    t.dtype()
                ));
            }
        }
        if b.shape() != a.shape() || b.shape() != beta_out.shape() || b.shape() != g_out.shape() {
            return Err(anyhow!(
                "compute_g_full: b/a/beta_out/g_out shape mismatch: {:?} {:?} {:?} {:?}",
                b.shape(),
                a.shape(),
                beta_out.shape(),
                g_out.shape(),
            ));
        }
        let n: usize = b.shape().iter().product();
        if n == 0 {
            return Ok(());
        }
        let hv = *b.shape().last().unwrap_or(&1);
        if dt_bias.shape() != [hv] || exp_a_log.shape() != [hv] {
            return Err(anyhow!(
                "compute_g_full: dt_bias/exp_a_log must be [{hv}], got {:?} / {:?}",
                dt_bias.shape(),
                exp_a_log.shape(),
            ));
        }

        enc.set_compute_pipeline_state(&self.compute_g_full_f32);
        enc.set_buffer(0, Some(b.buffer()), b.offset_bytes());
        enc.set_buffer(1, Some(a.buffer()), a.offset_bytes());
        enc.set_buffer(2, Some(dt_bias.buffer()), dt_bias.offset_bytes());
        enc.set_buffer(3, Some(exp_a_log.buffer()), exp_a_log.offset_bytes());
        enc.set_buffer(4, Some(beta_out.buffer()), beta_out.offset_bytes());
        enc.set_buffer(5, Some(g_out.buffer()), g_out.offset_bytes());
        let n_u32 = n as u32;
        let hv_u32 = hv as u32;
        enc.set_bytes_directly(6, 4, &n_u32 as *const _ as *const _);
        enc.set_bytes_directly(7, 4, &hv_u32 as *const _ as *const _);

        let max_threads = self.compute_g_full_f32.max_total_threads_per_threadgroup();
        let tg = MTLSize {
            width: max_threads.min(n).max(1),
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }

    /// Encode-only fused `y = silu(z) * x`. Used by the linear-attention
    /// output path to keep RMSNormGated inside the post-conv command buffer
    /// (Phase A.8-D), so out_proj can be encoded against `y` without an
    /// intermediate commit.
    ///
    /// All three tensors must share the same f32 shape (typically
    /// `[B, S, V]` flattened over heads × head_dim).
    pub fn encode_silu_mul(
        &self,
        enc: &ComputeCommandEncoderRef,
        z: &NativeTensor,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_binary_fused(enc, &self.silu_mul_f32, z, x, y, "silu_mul")
    }

    /// Encode-only fused `y = sigmoid(g) * x`. Used by the self-attention
    /// output gate to keep gating + o_proj inside the post-attention command
    /// buffer (Phase A.8-D).
    pub fn encode_sigmoid_mul(
        &self,
        enc: &ComputeCommandEncoderRef,
        g: &NativeTensor,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        self.encode_binary_fused(enc, &self.sigmoid_mul_f32, g, x, y, "sigmoid_mul")
    }

    /// Internal helper for two-input → one-output element-wise kernels with
    /// the `(a, b, y, n)` argument layout. Validates dtype + matching shapes,
    /// no-ops on empty tensors, and dispatches a 1D grid sized to `numel`.
    fn encode_binary_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        a: &NativeTensor,
        b: &NativeTensor,
        y: &NativeTensor,
        op_name: &str,
    ) -> Result<()> {
        if a.dtype() != NativeDType::F32
            || b.dtype() != NativeDType::F32
            || y.dtype() != NativeDType::F32
        {
            return Err(anyhow!("{op_name} f32 only"));
        }
        if a.shape() != b.shape() || a.shape() != y.shape() {
            return Err(anyhow!(
                "{op_name} shape mismatch: a={:?} b={:?} y={:?}",
                a.shape(),
                b.shape(),
                y.shape(),
            ));
        }
        let n = a.numel();
        if n == 0 {
            return Ok(());
        }
        let max_threads = pso.max_total_threads_per_threadgroup();
        let tg_w = n.min(256).min(max_threads).max(1);

        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(a.buffer()), a.offset_bytes());
        enc.set_buffer(1, Some(b.buffer()), b.offset_bytes());
        enc.set_buffer(2, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(3, 4, &n_u32 as *const _ as *const _);

        let grid = MTLSize {
            width: n as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }
}

// ─── Linear-attn helpers (Phase A.8-C.3) ────────────────────────────────────
impl KernelLib {
    /// `y = x + bias[h]` where the trailing axis of `x` is the per-head dim.
    /// `x` is logically `[B, S, Hv]` (rank 2 or 3 accepted; only the trailing
    /// dim must equal `bias.len()`).
    pub fn broadcast_add_per_head(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        bias: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_broadcast_per_head(x, bias, y, "broadcast_add_per_head")?;
        if prep.is_none() {
            return Ok(());
        }
        let (n, hv, tg_w) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:broadcast_add_per_head");
        self.dispatch_broadcast_per_head(
            &enc,
            &self.broadcast_add_per_head_f32,
            x,
            bias,
            y,
            n,
            hv,
            tg_w,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only [`Self::broadcast_add_per_head`].
    pub fn encode_broadcast_add_per_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        bias: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((n, hv, tg_w)) =
            self.validate_broadcast_per_head(x, bias, y, "broadcast_add_per_head")?
        {
            self.dispatch_broadcast_per_head(
                enc,
                &self.broadcast_add_per_head_f32,
                x,
                bias,
                y,
                n,
                hv,
                tg_w,
            );
        }
        Ok(())
    }

    /// `y = x * scale[h]` with per-head broadcast on the trailing axis.
    pub fn mul_broadcast_per_head(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        scale: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_broadcast_per_head(x, scale, y, "mul_broadcast_per_head")?;
        if prep.is_none() {
            return Ok(());
        }
        let (n, hv, tg_w) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:mul_broadcast_per_head");
        self.dispatch_broadcast_per_head(
            &enc,
            &self.mul_broadcast_per_head_f32,
            x,
            scale,
            y,
            n,
            hv,
            tg_w,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only [`Self::mul_broadcast_per_head`].
    pub fn encode_mul_broadcast_per_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        scale: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((n, hv, tg_w)) =
            self.validate_broadcast_per_head(x, scale, y, "mul_broadcast_per_head")?
        {
            self.dispatch_broadcast_per_head(
                enc,
                &self.mul_broadcast_per_head_f32,
                x,
                scale,
                y,
                n,
                hv,
                tg_w,
            );
        }
        Ok(())
    }

    fn validate_broadcast_per_head(
        &self,
        x: &NativeTensor,
        b: &NativeTensor,
        y: &NativeTensor,
        op: &str,
    ) -> Result<Option<(usize, usize, usize)>> {
        if x.dtype() != NativeDType::F32
            || b.dtype() != NativeDType::F32
            || y.dtype() != NativeDType::F32
        {
            return Err(anyhow!("{op} f32 only"));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "{op} x/y shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        if x.rank() < 1 {
            return Err(anyhow!("{op} requires rank >= 1"));
        }
        let hv = *x.shape().last().unwrap();
        if b.rank() != 1 || b.shape()[0] != hv {
            return Err(anyhow!(
                "{op} bias/scale must be rank-1 [{hv}], got {:?}",
                b.shape()
            ));
        }
        let n = x.numel();
        if n == 0 || hv == 0 {
            return Ok(None);
        }
        let tg_w = (n as usize).min(256).max(1);
        Ok(Some((n, hv, tg_w)))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_broadcast_per_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        x: &NativeTensor,
        b: &NativeTensor,
        y: &NativeTensor,
        n: usize,
        hv: usize,
        tg_w_hint: usize,
    ) {
        let max_threads = pso.max_total_threads_per_threadgroup();
        let tg_w = tg_w_hint.min(max_threads).max(1);
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(b.buffer()), b.offset_bytes());
        enc.set_buffer(2, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        let hv_u32 = hv as u32;
        enc.set_bytes_directly(3, 4, &n_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &hv_u32 as *const _ as *const _);

        let grid = MTLSize {
            width: n as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }

    /// `y = scale * x + bias` element-wise. Replaces Candle's `Tensor::affine`
    /// for hot-path scalar rescales (e.g. the two `inv_scale` multiplies after
    /// the SSM Q/K weightless RMSNorm).
    pub fn affine_scalar(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        scale: f32,
        bias: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = Self::validate_elementwise(x, y, "affine_scalar")?;
        if prep.is_none() {
            return Ok(());
        }
        let (n, tg_w) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:affine_scalar");
        self.dispatch_affine_scalar(&enc, x, scale, bias, y, n, tg_w);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only [`Self::affine_scalar`].
    pub fn encode_affine_scalar(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        scale: f32,
        bias: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((n, tg_w)) = Self::validate_elementwise(x, y, "affine_scalar")? {
            self.dispatch_affine_scalar(&enc, x, scale, bias, y, n, tg_w);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_affine_scalar(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        scale: f32,
        bias: f32,
        y: &NativeTensor,
        n: usize,
        tg_w_hint: usize,
    ) {
        let max_threads = self.affine_scalar_f32.max_total_threads_per_threadgroup();
        let tg_w = tg_w_hint.min(max_threads).max(1);
        enc.set_compute_pipeline_state(&self.affine_scalar_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(2, 4, &n_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &scale as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &bias as *const _ as *const _);

        let grid = MTLSize {
            width: n as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }

    /// Depthwise causal conv1d + fused SiLU. Replaces the GatedDeltaNet
    /// 8-op Candle path (4 narrows → stack → broadcast_mul → sum → silu)
    /// with a single dispatch.
    ///
    /// Shapes (all F32, contiguous):
    ///   - `x`: `[B, kernel_size - 1 + S, C]` (prev_conv_state ++ qkv_flat)
    ///   - `w`: `[C, kernel_size]` (depthwise per-channel weights)
    ///   - `y`: `[B, S, C]` (post-SiLU output, pre-allocated)
    pub fn depthwise_conv1d_silu(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        w: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = self.validate_depthwise_conv1d(x, w, y)?;
        if prep.is_none() {
            return Ok(());
        }
        let (b, s, k, c) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:depthwise_conv1d_silu");
        self.dispatch_depthwise_conv1d_silu(&enc, x, w, y, b, s, k, c);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only [`Self::depthwise_conv1d_silu`].
    pub fn encode_depthwise_conv1d_silu(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        w: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if let Some((b, s, k, c)) = self.validate_depthwise_conv1d(x, w, y)? {
            self.dispatch_depthwise_conv1d_silu(enc, x, w, y, b, s, k, c);
        }
        Ok(())
    }

    fn validate_depthwise_conv1d(
        &self,
        x: &NativeTensor,
        w: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<Option<(usize, usize, usize, usize)>> {
        for (name, t) in [("x", x), ("w", w), ("y", y)] {
            if t.dtype() != NativeDType::F32 {
                return Err(anyhow!(
                    "depthwise_conv1d {name} dtype {:?} != F32",
                    t.dtype()
                ));
            }
        }
        if x.rank() != 3 || y.rank() != 3 {
            return Err(anyhow!(
                "depthwise_conv1d: x/y must be rank 3 (got x.rank={} y.rank={})",
                x.rank(),
                y.rank()
            ));
        }
        if w.rank() != 2 {
            return Err(anyhow!(
                "depthwise_conv1d: w must be rank 2 [C, K], got rank {}",
                w.rank()
            ));
        }
        let (b, x_total, c_x) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (b_y, s, c_y) = (y.shape()[0], y.shape()[1], y.shape()[2]);
        let (c_w, k) = (w.shape()[0], w.shape()[1]);
        if b != b_y {
            return Err(anyhow!("depthwise_conv1d: batch mismatch x={b} y={b_y}"));
        }
        if c_x != c_y || c_x != c_w {
            return Err(anyhow!(
                "depthwise_conv1d: channel mismatch x={c_x} y={c_y} w={c_w}"
            ));
        }
        if x_total != k - 1 + s {
            return Err(anyhow!(
                "depthwise_conv1d: x time-dim {x_total} != kernel-1 + seq ({} + {s})",
                k - 1
            ));
        }
        if b == 0 || s == 0 || c_x == 0 || k == 0 {
            return Ok(None);
        }
        Ok(Some((b, s, k, c_x)))
    }

    fn dispatch_depthwise_conv1d_silu(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        w: &NativeTensor,
        y: &NativeTensor,
        batch: usize,
        seq: usize,
        ksize: usize,
        chan: usize,
    ) {
        enc.set_compute_pipeline_state(&self.depthwise_conv1d_silu_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(w.buffer()), w.offset_bytes());
        enc.set_buffer(2, Some(y.buffer()), y.offset_bytes());
        let b_u32 = batch as u32;
        let s_u32 = seq as u32;
        let k_u32 = ksize as u32;
        let c_u32 = chan as u32;
        enc.set_bytes_directly(3, 4, &b_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &s_u32 as *const _ as *const _);
        enc.set_bytes_directly(5, 4, &k_u32 as *const _ as *const _);
        enc.set_bytes_directly(6, 4, &c_u32 as *const _ as *const _);

        let grid = MTLSize {
            width: chan,
            height: seq,
            depth: batch,
        };
        // Threadgroup width: cap at min(chan, 256) so small-C cases still
        // fit a threadgroup; depth-wise channels stride along x-dim.
        let max_threads = self
            .depthwise_conv1d_silu_f32
            .max_total_threads_per_threadgroup();
        let tg_w = chan.min(max_threads).min(256).max(1);
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }

    /// Repeat each head `repeats` times along axis 2 of a `[B, S, Hk, D]`
    /// tensor (`repeat_interleave` semantics): output head index `hv` reads
    /// from source head `hv / repeats`.
    pub fn repeat_heads_blhd(
        &self,
        ctx: &NativeContext,
        x: &NativeTensor,
        y: &NativeTensor,
        repeats: usize,
    ) -> Result<()> {
        let prep = self.validate_repeat_heads(x, y, repeats)?;
        if prep.is_none() {
            return Ok(());
        }
        let (b, s, hk, head_dim) = prep.unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("lumen:repeat_heads_blhd");
        self.dispatch_repeat_heads(&enc, x, y, b, s, hk, repeats, head_dim);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Encode-only [`Self::repeat_heads_blhd`].
    pub fn encode_repeat_heads_blhd(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
        repeats: usize,
    ) -> Result<()> {
        if let Some((b, s, hk, head_dim)) = self.validate_repeat_heads(x, y, repeats)? {
            self.dispatch_repeat_heads(&enc, x, y, b, s, hk, repeats, head_dim);
        }
        Ok(())
    }

    fn validate_repeat_heads(
        &self,
        x: &NativeTensor,
        y: &NativeTensor,
        repeats: usize,
    ) -> Result<Option<(usize, usize, usize, usize)>> {
        if x.dtype() != NativeDType::F32 || y.dtype() != NativeDType::F32 {
            return Err(anyhow!("repeat_heads_blhd f32 only"));
        }
        if x.rank() != 4 || y.rank() != 4 {
            return Err(anyhow!(
                "repeat_heads_blhd needs rank-4 tensors, got x={:?} y={:?}",
                x.shape(),
                y.shape()
            ));
        }
        if repeats == 0 {
            return Err(anyhow!("repeat_heads_blhd repeats must be > 0"));
        }
        let (b, s, hk, head_dim) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let hv = hk * repeats;
        if y.shape() != [b, s, hv, head_dim] {
            return Err(anyhow!(
                "repeat_heads_blhd y shape {:?} != [{b}, {s}, {hv}, {head_dim}]",
                y.shape()
            ));
        }
        if b == 0 || s == 0 || hk == 0 || head_dim == 0 {
            return Ok(None);
        }
        Ok(Some((b, s, hk, head_dim)))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_repeat_heads(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
        b: usize,
        s: usize,
        hk: usize,
        repeats: usize,
        head_dim: usize,
    ) {
        enc.set_compute_pipeline_state(&self.repeat_heads_blhd_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let hk_u32 = hk as u32;
        let rep_u32 = repeats as u32;
        let dh_u32 = head_dim as u32;
        enc.set_bytes_directly(2, 4, &hk_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &rep_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &dh_u32 as *const _ as *const _);

        let max_threads = self
            .repeat_heads_blhd_f32
            .max_total_threads_per_threadgroup();
        let tx = max_threads.min(head_dim as usize).max(1);
        let hv = hk * repeats;
        let grid = MTLSize {
            width: head_dim as usize,
            height: hv as usize,
            depth: (b * s) as usize,
        };
        let tg = MTLSize {
            width: tx,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    }
}

// ─── Workstream B Phase 6 — bf16 SSM subgraph encode helpers ────────────────
//
// `encode_*_bf16` mirror their f32 cousins above. All bf16 buffers must use
// `NativeDType::BF16`. State buffers in `encode_ssm_step_bf16` stay F32
// (Escape #3, recurrent state). Validation enforces these dtype contracts so
// caller mistakes fail loudly rather than silently miscomputing.
impl KernelLib {
    pub fn encode_rms_norm_weightless_bf16(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        eps: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        let prep = Self::validate_rms_norm_weightless_bf16(x, y)?;
        if prep.is_none() {
            return Ok(());
        }
        let (rows, hidden, tg_size) = prep.unwrap();
        enc.set_compute_pipeline_state(&self.rms_norm_weightless_bf16);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let hidden_u32 = hidden as u32;
        enc.set_bytes_directly(2, 4, &hidden_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &eps as *const _ as *const _);
        // Reduction scratch is float (we widen bf16 reads inside the kernel).
        enc.set_threadgroup_memory_length(0, tg_size * 4);
        let grid_groups = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
        Ok(())
    }

    fn validate_rms_norm_weightless_bf16(
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<Option<(usize, usize, usize)>> {
        if x.dtype() != NativeDType::BF16 || y.dtype() != NativeDType::BF16 {
            return Err(anyhow!("rms_norm_weightless_bf16: x/y must be BF16"));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "rms_norm_weightless_bf16 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        if x.rank() < 1 {
            return Err(anyhow!("rms_norm_weightless_bf16 requires rank >= 1"));
        }
        let hidden = *x.shape().last().unwrap();
        let rows: usize = x.shape().iter().rev().skip(1).product();
        if rows == 0 || hidden == 0 {
            return Ok(None);
        }
        // Power-of-2 threadgroup width ≤ min(hidden, 256). Match the f32
        // weightless variant's geometry.
        let mut tg_size: usize = 1;
        while tg_size * 2 <= hidden.min(256) {
            tg_size *= 2;
        }
        if tg_size == 0 {
            tg_size = 1;
        }
        Ok(Some((rows, hidden, tg_size)))
    }

    pub fn encode_affine_scalar_bf16(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        scale: f32,
        bias: f32,
        y: &NativeTensor,
    ) -> Result<()> {
        if x.dtype() != NativeDType::BF16 || y.dtype() != NativeDType::BF16 {
            return Err(anyhow!("affine_scalar_bf16: x/y must be BF16"));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "affine_scalar_bf16 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let n = x.numel();
        if n == 0 {
            return Ok(());
        }
        let max_threads = self.affine_scalar_bf16.max_total_threads_per_threadgroup();
        let tg_w = n.min(max_threads).min(256).max(1);
        enc.set_compute_pipeline_state(&self.affine_scalar_bf16);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(2, 4, &n_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &scale as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &bias as *const _ as *const _);
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }

    pub fn encode_repeat_heads_blhd_bf16(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
        repeats: usize,
    ) -> Result<()> {
        if x.dtype() != NativeDType::BF16 || y.dtype() != NativeDType::BF16 {
            return Err(anyhow!("repeat_heads_blhd_bf16: x/y must be BF16"));
        }
        if x.rank() != 4 || y.rank() != 4 {
            return Err(anyhow!(
                "repeat_heads_blhd_bf16 needs rank-4 tensors, got x={:?} y={:?}",
                x.shape(),
                y.shape()
            ));
        }
        if repeats == 0 {
            return Err(anyhow!("repeat_heads_blhd_bf16 repeats must be > 0"));
        }
        let (b, s, hk, head_dim) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let hv = hk * repeats;
        if y.shape() != [b, s, hv, head_dim] {
            return Err(anyhow!(
                "repeat_heads_blhd_bf16 y shape {:?} != [{b}, {s}, {hv}, {head_dim}]",
                y.shape()
            ));
        }
        if b == 0 || s == 0 || hk == 0 || head_dim == 0 {
            return Ok(());
        }
        enc.set_compute_pipeline_state(&self.repeat_heads_blhd_bf16);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let hk_u32 = hk as u32;
        let rep_u32 = repeats as u32;
        let dh_u32 = head_dim as u32;
        enc.set_bytes_directly(2, 4, &hk_u32 as *const _ as *const _);
        enc.set_bytes_directly(3, 4, &rep_u32 as *const _ as *const _);
        enc.set_bytes_directly(4, 4, &dh_u32 as *const _ as *const _);
        let max_threads = self
            .repeat_heads_blhd_bf16
            .max_total_threads_per_threadgroup();
        let tx = max_threads.min(head_dim).max(1);
        let grid = MTLSize {
            width: head_dim,
            height: hv,
            depth: b * s,
        };
        let tg = MTLSize {
            width: tx,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }

    /// bf16 SSM step. Same dispatch geometry as the f32 variant; q/k/v/beta/g
    /// are bf16 inputs, y is bf16 output, state stays f32 (Escape #3 — the
    /// recurrent SSM state cannot live in bf16 without drift across long
    /// decodes; this matches MLX's `_gated_delta_step_ops` and the
    /// `tq_gated_delta_step_bf16` ops-path kernel in lumen-metal).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ssm_step_bf16(
        &self,
        enc: &ComputeCommandEncoderRef,
        state: &NativeTensor,
        q: &NativeTensor,
        k: &NativeTensor,
        v: &NativeTensor,
        beta: &NativeTensor,
        g: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if state.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "ssm_step_bf16: state must be F32 (Escape #3), got {:?}",
                state.dtype()
            ));
        }
        for (name, t) in [
            ("q", q),
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("g", g),
            ("y", y),
        ] {
            if t.dtype() != NativeDType::BF16 {
                return Err(anyhow!(
                    "ssm_step_bf16: {name} dtype {:?} != BF16",
                    t.dtype()
                ));
            }
        }
        if state.rank() != 4 {
            return Err(anyhow!(
                "ssm_step_bf16 state must be rank 4 [B,Hv,Dv,Dk], got {:?}",
                state.shape()
            ));
        }
        let (b, hv, dv, dk) = (
            state.shape()[0],
            state.shape()[1],
            state.shape()[2],
            state.shape()[3],
        );
        if q.shape() != [b, hv, dk] || k.shape() != [b, hv, dk] {
            return Err(anyhow!(
                "ssm_step_bf16 q/k must be [{b},{hv},{dk}], got q={:?} k={:?}",
                q.shape(),
                k.shape()
            ));
        }
        if v.shape() != [b, hv, dv] || y.shape() != [b, hv, dv] {
            return Err(anyhow!(
                "ssm_step_bf16 v/y must be [{b},{hv},{dv}], got v={:?} y={:?}",
                v.shape(),
                y.shape()
            ));
        }
        if beta.shape() != [b, hv] || g.shape() != [b, hv] {
            return Err(anyhow!(
                "ssm_step_bf16 beta/g must be [{b},{hv}], got beta={:?} g={:?}",
                beta.shape(),
                g.shape()
            ));
        }
        if b == 0 || hv == 0 || dv == 0 || dk == 0 {
            return Ok(());
        }
        let max_threads = self.ssm_step_bf16.max_total_threads_per_threadgroup();
        let want = dk.min(max_threads).min(256);
        let mut tg_size: usize = 1;
        while tg_size * 2 <= want {
            tg_size *= 2;
        }
        if tg_size == 0 {
            tg_size = 1;
        }

        enc.set_compute_pipeline_state(&self.ssm_step_bf16);
        enc.set_buffer(0, Some(state.buffer()), state.offset_bytes());
        enc.set_buffer(1, Some(q.buffer()), q.offset_bytes());
        enc.set_buffer(2, Some(k.buffer()), k.offset_bytes());
        enc.set_buffer(3, Some(v.buffer()), v.offset_bytes());
        enc.set_buffer(4, Some(beta.buffer()), beta.offset_bytes());
        enc.set_buffer(5, Some(g.buffer()), g.offset_bytes());
        enc.set_buffer(6, Some(y.buffer()), y.offset_bytes());
        let hv_u32 = hv as u32;
        let dv_u32 = dv as u32;
        let dk_u32 = dk as u32;
        let tg_u32 = tg_size as u32;
        enc.set_bytes_directly(7, 4, &hv_u32 as *const _ as *const _);
        enc.set_bytes_directly(8, 4, &dv_u32 as *const _ as *const _);
        enc.set_bytes_directly(9, 4, &dk_u32 as *const _ as *const _);
        enc.set_bytes_directly(10, 4, &tg_u32 as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, tg_size * 4);

        let grid_groups = MTLSize {
            width: dv,
            height: hv,
            depth: b,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid_groups, tg);
        Ok(())
    }

    /// Element-wise f32 → bf16 cast. Used to bridge `beta`/`g` (computed in
    /// f32 by the sigmoid/softplus/exp chain on the host queue) into bf16
    /// before [`Self::encode_ssm_step_bf16`]. Cheap: `B*S*Hv` elements per
    /// layer, dominated by the kernel launch fixed cost.
    pub fn encode_cast_f32_to_bf16(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if x.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "cast_f32_to_bf16: x must be F32, got {:?}",
                x.dtype()
            ));
        }
        if y.dtype() != NativeDType::BF16 {
            return Err(anyhow!(
                "cast_f32_to_bf16: y must be BF16, got {:?}",
                y.dtype()
            ));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "cast_f32_to_bf16 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let n = x.numel();
        if n == 0 {
            return Ok(());
        }
        let max_threads = self.cast_f32_to_bf16.max_total_threads_per_threadgroup();
        let tg_w = n.min(max_threads).min(256).max(1);
        enc.set_compute_pipeline_state(&self.cast_f32_to_bf16);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(2, 4, &n_u32 as *const _ as *const _);
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }

    /// Element-wise bf16 → f32 cast. Used to bridge `y_n` (bf16 output of
    /// [`Self::encode_ssm_step_bf16`]) into the f32 RMSNormGated tail.
    pub fn encode_cast_bf16_to_f32(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &NativeTensor,
        y: &NativeTensor,
    ) -> Result<()> {
        if x.dtype() != NativeDType::BF16 {
            return Err(anyhow!(
                "cast_bf16_to_f32: x must be BF16, got {:?}",
                x.dtype()
            ));
        }
        if y.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "cast_bf16_to_f32: y must be F32, got {:?}",
                y.dtype()
            ));
        }
        if x.shape() != y.shape() {
            return Err(anyhow!(
                "cast_bf16_to_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            ));
        }
        let n = x.numel();
        if n == 0 {
            return Ok(());
        }
        let max_threads = self.cast_bf16_to_f32.max_total_threads_per_threadgroup();
        let tg_w = n.min(max_threads).min(256).max(1);
        enc.set_compute_pipeline_state(&self.cast_bf16_to_f32);
        enc.set_buffer(0, Some(x.buffer()), x.offset_bytes());
        enc.set_buffer(1, Some(y.buffer()), y.offset_bytes());
        let n_u32 = n as u32;
        enc.set_bytes_directly(2, 4, &n_u32 as *const _ as *const _);
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        Ok(())
    }
}

/// Build the (cos, sin) tables matching `qwen3_5_moe::self_attn::build_rope_table`.
///
/// - `freqs[i] = base^(2i / rotary_dim)` for i in [0, half)
/// - `angle(t, i) = (t + pos_offset) / freqs[i]`
/// - `cos[t, i] = cos(angle)`, `sin[t, i] = sin(angle)`
///
/// Returns two `NativeTensor`s of shape `[seq_len, rotary_dim/2]` ready to feed
/// the `rope_partial` kernel.
pub fn build_rope_tables(
    ctx: &NativeContext,
    rotary_dim: usize,
    seq_len: usize,
    pos_offset: usize,
    theta: f32,
) -> Result<(NativeTensor, NativeTensor)> {
    if rotary_dim % 2 != 0 {
        return Err(anyhow!("rotary_dim {rotary_dim} must be even"));
    }
    let half = rotary_dim / 2;
    let inv_freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2 * i) as f32 / rotary_dim as f32;
            1.0 / theta.powf(exp)
        })
        .collect();
    let mut cos_v = vec![0.0_f32; seq_len * half];
    let mut sin_v = vec![0.0_f32; seq_len * half];
    for t in 0..seq_len {
        let pos = (pos_offset + t) as f32;
        for i in 0..half {
            let a = pos * inv_freqs[i];
            cos_v[t * half + i] = a.cos();
            sin_v[t * half + i] = a.sin();
        }
    }
    let cos = ctx.from_slice_f32(&cos_v, vec![seq_len, half])?;
    let sin = ctx.from_slice_f32(&sin_v, vec![seq_len, half])?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup_matches_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let vocab = 5;
        let hidden = 8;
        let table: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.25).collect();
        let table_t = ctx.from_slice_f32(&table, vec![vocab, hidden]).unwrap();
        let token_ids: Vec<u32> = vec![3, 0, 4, 2];
        let mut tok_buf = ctx.uninit(vec![token_ids.len()], NativeDType::U32).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                token_ids.as_ptr(),
                tok_buf.buffer().contents() as *mut u32,
                token_ids.len(),
            );
        }
        // No-op to silence "mut" (the buffer view is mutated through raw pointer).
        let _ = &mut tok_buf;
        let out_t = ctx
            .zeros(vec![token_ids.len(), hidden], NativeDType::F32)
            .unwrap();

        lib.embedding_lookup(&ctx, &tok_buf, &table_t, &out_t)
            .unwrap();

        let got = out_t.to_vec_f32().unwrap();
        let mut expected = Vec::with_capacity(token_ids.len() * hidden);
        for id in &token_ids {
            for h in 0..hidden {
                expected.push(table[*id as usize * hidden + h]);
            }
        }
        assert_eq!(got, expected);
    }

    fn cpu_rms_norm(x: &[f32], gamma: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
        let rows = x.len() / hidden;
        let mut out = vec![0.0_f32; x.len()];
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / (hidden as f32);
            let scale = (mean_sq + eps).sqrt().recip();
            for (i, v) in row.iter().enumerate() {
                out[r * hidden + i] = v * scale * gamma[i];
            }
        }
        out
    }

    fn cpu_rope_partial(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut out = x.to_vec();
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    let base = ((b * seq + s) * heads + h) * head_dim;
                    for i in 0..half {
                        let c = cos[s * half + i];
                        let si = sin[s * half + i];
                        let a = x[base + i];
                        let bv = x[base + i + half];
                        out[base + i] = a * c - bv * si;
                        out[base + i + half] = a * si + bv * c;
                    }
                    for i in rotary_dim..head_dim {
                        out[base + i] = x[base + i];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn rope_partial_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let batch = 2;
        let seq = 4;
        let heads = 3;
        let head_dim = 16;
        let rotary_dim = 8; // partial: half = 4
        let half = rotary_dim / 2;
        let theta = 10_000.0_f32;

        let x: Vec<f32> = (0..batch * seq * heads * head_dim)
            .map(|i| ((i as f32) * 0.017).sin() * 1.3 + ((i as f32) * 0.003).cos())
            .collect();

        // Build cos/sin via builder so the kernel exercises the same path
        // production callers use.
        let (cos_t, sin_t) = build_rope_tables(&ctx, rotary_dim, seq, 0, theta).unwrap();
        let cos_v = cos_t.to_vec_f32().unwrap();
        let sin_v = sin_t.to_vec_f32().unwrap();
        assert_eq!(cos_v.len(), seq * half);

        let x_t = ctx
            .from_slice_f32(&x, vec![batch, seq, heads, head_dim])
            .unwrap();
        let y_t = ctx
            .zeros(vec![batch, seq, heads, head_dim], NativeDType::F32)
            .unwrap();

        lib.rope_partial(&ctx, &x_t, &cos_t, &sin_t, &y_t, half)
            .unwrap();

        let got = y_t.to_vec_f32().unwrap();
        let expected =
            cpu_rope_partial(&x, &cos_v, &sin_v, batch, seq, heads, head_dim, rotary_dim);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "rope idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    #[test]
    fn rope_partial_matches_candle_rotary_emb() {
        // Cross-check against Candle's `rotary_emb::rope` to confirm convention
        // alignment (GPT-NeoX split form). When Candle's reference is unavailable
        // (CPU-only build), this test is a no-op.
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let batch = 1;
        let seq = 6;
        let heads = 4;
        let head_dim = 32;
        let rotary_dim = 32; // full rotary in this test
        let half = rotary_dim / 2;
        let theta = 1_000_000.0_f32;

        let x: Vec<f32> = (0..batch * seq * heads * head_dim)
            .map(|i| (((i as f32) * 0.009).sin() + 0.1 * (i as f32 * 0.01).cos()))
            .collect();

        let (cos_t, sin_t) = build_rope_tables(&ctx, rotary_dim, seq, 0, theta).unwrap();
        let cos_v = cos_t.to_vec_f32().unwrap();
        let sin_v = sin_t.to_vec_f32().unwrap();

        let x_t = ctx
            .from_slice_f32(&x, vec![batch, seq, heads, head_dim])
            .unwrap();
        let y_t = ctx
            .zeros(vec![batch, seq, heads, head_dim], NativeDType::F32)
            .unwrap();
        lib.rope_partial(&ctx, &x_t, &cos_t, &sin_t, &y_t, half)
            .unwrap();
        let got = y_t.to_vec_f32().unwrap();

        // Candle reference. `rotary_emb::rope` expects [B, H, L, D] with cos/sin
        // shaped [L, D/2]. We feed it the same data and reshape.
        use candle_core::{Device, Tensor};
        use candle_nn::rotary_emb;
        let device = Device::Cpu;
        let x_candle = Tensor::from_vec(x.clone(), (batch, seq, heads, head_dim), &device)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let cos_c = Tensor::from_vec(cos_v.clone(), (seq, half), &device).unwrap();
        let sin_c = Tensor::from_vec(sin_v.clone(), (seq, half), &device).unwrap();
        let y_candle = rotary_emb::rope(&x_candle, &cos_c, &sin_c).unwrap();
        // Back to [B, L, H, D] for comparison.
        let y_candle = y_candle.transpose(1, 2).unwrap().contiguous().unwrap();
        let expected = y_candle.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "candle vs native rope idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    /// Production-size attention_causal: B=1, q_heads=16, kv_heads=2, l=11, head_dim=256.
    #[test]
    fn attention_causal_at_production_size() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 1usize;
        let q_heads = 16;
        let kv_heads = 2;
        let l = 11usize;
        let head_dim = 256usize;
        let pos_offset = 0usize;

        let make = |seed: u32, n: usize| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.4
                })
                .collect()
        };
        let q_v = make(0x10, b * q_heads * l * head_dim);
        let k_v = make(0x20, b * kv_heads * l * head_dim);
        let v_v = make(0x30, b * kv_heads * l * head_dim);

        let q = ctx
            .from_slice_f32(&q_v, vec![b, q_heads, l, head_dim])
            .unwrap();
        let k = ctx
            .from_slice_f32(&k_v, vec![b, kv_heads, l, head_dim])
            .unwrap();
        let v = ctx
            .from_slice_f32(&v_v, vec![b, kv_heads, l, head_dim])
            .unwrap();
        let out = ctx
            .zeros(vec![b, q_heads, l, head_dim], NativeDType::F32)
            .unwrap();
        lib.attention_causal(&ctx, &q, &k, &v, &out, pos_offset)
            .unwrap();
        let got = out.to_vec_f32().unwrap();

        let expected = cpu_attention(
            &q_v, &k_v, &v_v, b, q_heads, kv_heads, l, l, head_dim, pos_offset,
        );
        let mut max_abs = 0.0_f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let err = (g - e).abs();
            if err > max_abs {
                max_abs = err;
            }
        }
        let dot: f64 = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = got.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = expected
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!("attention_causal production-size: cos={cos:.6} max_abs={max_abs:.6}");
        assert!(
            cos > 0.9999,
            "attention_causal production-size cos {cos} max_abs {max_abs}"
        );
    }

    /// Production-size weighted RMSNorm: head_dim=256, rows=176 (B*L*H = 1*11*16).
    /// Native rms_norm (with gamma) vs candle's `ops::rms_norm`. If diverges, that's
    /// the source of the layer-3 cos=0.68 issue.
    #[test]
    fn rms_norm_matches_candle_at_production_size() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let rows = 176;
        let hidden = 256;
        let eps = 1e-6_f32;

        let x: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.5)
            .collect();
        let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.005).collect();

        let x_n = ctx.from_slice_f32(&x, vec![rows, hidden]).unwrap();
        let gamma_n = ctx.from_slice_f32(&gamma, vec![hidden]).unwrap();
        let y_n = ctx.zeros(vec![rows, hidden], NativeDType::F32).unwrap();
        lib.rms_norm(&ctx, &x_n, &gamma_n, eps, &y_n).unwrap();
        let got = y_n.to_vec_f32().unwrap();

        // Candle reference: ops::rms_norm.
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;
        let x_c = Tensor::from_vec(x.clone(), (rows, hidden), &device).unwrap();
        let g_c = Tensor::from_vec(gamma.clone(), (hidden,), &device).unwrap();
        let y_c = candle_nn::ops::rms_norm(&x_c, &g_c, eps).unwrap();
        let expected = y_c.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut max_abs = 0.0_f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let err = (g - e).abs();
            if err > max_abs {
                max_abs = err;
            }
        }
        let dot: f64 = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = got.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = expected
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!("rms_norm production-size: cos={cos:.6} max_abs={max_abs:.6}");
        assert!(
            cos > 0.9999,
            "rms_norm production-size cos {cos} max_abs {max_abs}"
        );
    }

    /// Production-size partial rope test: head_dim=256, rotary_dim=64, num_heads=16.
    /// The smaller `rope_partial_matches_candle_rotary_emb` test uses full rotary
    /// (head_dim=32, rotary_dim=32) so it can't catch a bug specific to the partial
    /// pass-through region (channels [rotary_dim..head_dim] should stay unchanged).
    #[test]
    fn rope_partial_matches_candle_at_production_size() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let batch = 1;
        let seq = 11;
        let heads = 16;
        let head_dim = 256;
        let rotary_dim = 64; // partial — last 192 channels pass through
        let half = rotary_dim / 2;
        let theta = 10_000_000.0_f32;
        let pos_offset = 0;

        let x: Vec<f32> = (0..batch * seq * heads * head_dim)
            .map(|i| ((i as f32) * 0.001).sin() * 0.5 + ((i as f32) * 0.013).cos() * 0.3)
            .collect();

        let (cos_t, sin_t) = build_rope_tables(&ctx, rotary_dim, seq, pos_offset, theta).unwrap();
        let cos_v = cos_t.to_vec_f32().unwrap();
        let sin_v = sin_t.to_vec_f32().unwrap();

        let x_t = ctx
            .from_slice_f32(&x, vec![batch, seq, heads, head_dim])
            .unwrap();
        let y_t = ctx
            .zeros(vec![batch, seq, heads, head_dim], NativeDType::F32)
            .unwrap();
        lib.rope_partial(&ctx, &x_t, &cos_t, &sin_t, &y_t, half)
            .unwrap();
        let got = y_t.to_vec_f32().unwrap();

        // Candle reference: same partial rope, on BHLD layout.
        use candle_core::{D, Device, Tensor};
        use candle_nn::rotary_emb;
        let device = Device::Cpu;
        let x_bhld = Tensor::from_vec(x.clone(), (batch, seq, heads, head_dim), &device)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let cos_c = Tensor::from_vec(cos_v.clone(), (seq, half), &device).unwrap();
        let sin_c = Tensor::from_vec(sin_v.clone(), (seq, half), &device).unwrap();
        // Apply rope to first rotary_dim channels, pass-through the rest.
        let rot = x_bhld
            .narrow(D::Minus1, 0, rotary_dim)
            .unwrap()
            .contiguous()
            .unwrap();
        let pass = x_bhld
            .narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)
            .unwrap();
        let rot = rotary_emb::rope(&rot, &cos_c, &sin_c).unwrap();
        let y_candle = Tensor::cat(&[&rot, &pass], D::Minus1).unwrap();
        // Back to BLHD.
        let y_candle = y_candle.transpose(1, 2).unwrap().contiguous().unwrap();
        let expected = y_candle.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut max_abs = 0.0_f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let err = (g - e).abs();
            if err > max_abs {
                max_abs = err;
            }
        }
        // cosine
        let dot: f64 = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = got.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = expected
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!("rope_partial production-size: cos={cos:.6} max_abs={max_abs:.6}");
        assert!(
            cos > 0.9999,
            "rope_partial production-size cos {cos} max_abs {max_abs}"
        );
    }

    fn cpu_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        q_heads: usize,
        kv_heads: usize,
        l_q: usize,
        l_kv: usize,
        d: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; b * q_heads * l_q * d];
        let scale = 1.0_f32 / (d as f32).sqrt();
        let group = q_heads / kv_heads;
        for bi in 0..b {
            for q_h in 0..q_heads {
                let kv_h = q_h / group;
                for qi in 0..l_q {
                    let q_pos = pos_offset + qi;
                    let mut scores = vec![0.0_f32; l_kv];
                    let mut max_score = f32::NEG_INFINITY;
                    for ki in 0..l_kv {
                        if ki > q_pos {
                            scores[ki] = f32::NEG_INFINITY;
                            continue;
                        }
                        let mut dot = 0.0_f32;
                        let q_off = ((bi * q_heads + q_h) * l_q + qi) * d;
                        let k_off = ((bi * kv_heads + kv_h) * l_kv + ki) * d;
                        for di in 0..d {
                            dot += q[q_off + di] * k[k_off + di];
                        }
                        scores[ki] = dot * scale;
                        if scores[ki] > max_score {
                            max_score = scores[ki];
                        }
                    }
                    let mut sum = 0.0_f32;
                    for ki in 0..l_kv {
                        scores[ki] = (scores[ki] - max_score).exp();
                        sum += scores[ki];
                    }
                    if sum == 0.0 {
                        sum = 1.0;
                    }
                    let inv_sum = 1.0 / sum;
                    for di in 0..d {
                        let mut o = 0.0_f32;
                        for ki in 0..l_kv {
                            let v_off = ((bi * kv_heads + kv_h) * l_kv + ki) * d;
                            o += scores[ki] * v[v_off + di];
                        }
                        let out_off = ((bi * q_heads + q_h) * l_q + qi) * d;
                        out[out_off + di] = o * inv_sum;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn attention_causal_matches_cpu_reference_decode() {
        // Decode: l_q=1, pos_offset attends past keys.
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 1;
        let h = 4;
        let l_q = 1;
        let l_kv = 32;
        let d = 16;
        let pos_offset = l_kv - 1;

        let make = |seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..b * h * l_kv * d)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * 1.5
                })
                .collect()
        };
        let q: Vec<f32> = (0..b * h * l_q * d)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let k = make(0xDEAD);
        let v = make(0xBEEF);

        let q_t = ctx.from_slice_f32(&q, vec![b, h, l_q, d]).unwrap();
        let k_t = ctx.from_slice_f32(&k, vec![b, h, l_kv, d]).unwrap();
        let v_t = ctx.from_slice_f32(&v, vec![b, h, l_kv, d]).unwrap();
        let o_t = ctx.zeros(vec![b, h, l_q, d], NativeDType::F32).unwrap();

        lib.attention_causal(&ctx, &q_t, &k_t, &v_t, &o_t, pos_offset)
            .unwrap();

        let got = o_t.to_vec_f32().unwrap();
        let expected = cpu_attention(&q, &k, &v, b, h, h, l_q, l_kv, d, pos_offset);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "attn(decode) idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    #[test]
    fn attention_causal_gqa() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        // q_heads = 8, kv_heads = 2 → group = 4. Mirrors Qwen3.5-MoE 16/2 ratio.
        let b = 1;
        let q_h = 8;
        let kv_h = 2;
        let l_q = 4;
        let l_kv = 4;
        let d = 16;
        let pos_offset = 0;

        let q: Vec<f32> = (0..b * q_h * l_q * d)
            .map(|i| ((i as f32) * 0.011).sin() * 1.2)
            .collect();
        let k: Vec<f32> = (0..b * kv_h * l_kv * d)
            .map(|i| ((i as f32) * 0.013 + 0.3).cos() * 1.1)
            .collect();
        let v: Vec<f32> = (0..b * kv_h * l_kv * d)
            .map(|i| (i as f32) * 0.007 - 0.2)
            .collect();

        let q_t = ctx.from_slice_f32(&q, vec![b, q_h, l_q, d]).unwrap();
        let k_t = ctx.from_slice_f32(&k, vec![b, kv_h, l_kv, d]).unwrap();
        let v_t = ctx.from_slice_f32(&v, vec![b, kv_h, l_kv, d]).unwrap();
        let o_t = ctx.zeros(vec![b, q_h, l_q, d], NativeDType::F32).unwrap();

        lib.attention_causal(&ctx, &q_t, &k_t, &v_t, &o_t, pos_offset)
            .unwrap();

        let got = o_t.to_vec_f32().unwrap();
        let expected = cpu_attention(&q, &k, &v, b, q_h, kv_h, l_q, l_kv, d, pos_offset);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "attn(GQA) idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    #[test]
    fn transpose_blhd_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 2;
        let l = 5;
        let h = 3;
        let d = 7;
        let data: Vec<f32> = (0..b * l * h * d).map(|i| i as f32 * 0.1).collect();
        let x = ctx.from_slice_f32(&data, vec![b, l, h, d]).unwrap();
        let y_bhld = ctx.zeros(vec![b, h, l, d], NativeDType::F32).unwrap();
        lib.transpose_blhd(&ctx, &x, &y_bhld, 0).unwrap();
        // Verify content: y_bhld[b, h, l, d] == data[b, l, h, d]
        let got = y_bhld.to_vec_f32().unwrap();
        for bi in 0..b {
            for hi in 0..h {
                for li in 0..l {
                    for di in 0..d {
                        let src = ((bi * l + li) * h + hi) * d + di;
                        let dst = ((bi * h + hi) * l + li) * d + di;
                        assert!(
                            (got[dst] - data[src]).abs() < 1e-6,
                            "transpose forward mismatch at b={bi} h={hi} l={li} d={di}"
                        );
                    }
                }
            }
        }
        // Roundtrip back to BLHD
        let x_back = ctx.zeros(vec![b, l, h, d], NativeDType::F32).unwrap();
        lib.transpose_blhd(&ctx, &y_bhld, &x_back, 1).unwrap();
        let back = x_back.to_vec_f32().unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn attention_causal_matches_cpu_reference_prefill() {
        // Prefill: l_q == l_kv, full causal triangle.
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 2;
        let h = 3;
        let l = 8;
        let d = 16;
        let pos_offset = 0;

        let q: Vec<f32> = (0..b * h * l * d)
            .map(|i| ((i as f32) * 0.013).sin() * 1.4)
            .collect();
        let k: Vec<f32> = (0..b * h * l * d)
            .map(|i| ((i as f32) * 0.017 + 0.5).cos() * 1.2)
            .collect();
        let v: Vec<f32> = (0..b * h * l * d)
            .map(|i| (i as f32) * 0.005 - 0.3)
            .collect();

        let q_t = ctx.from_slice_f32(&q, vec![b, h, l, d]).unwrap();
        let k_t = ctx.from_slice_f32(&k, vec![b, h, l, d]).unwrap();
        let v_t = ctx.from_slice_f32(&v, vec![b, h, l, d]).unwrap();
        let o_t = ctx.zeros(vec![b, h, l, d], NativeDType::F32).unwrap();

        lib.attention_causal(&ctx, &q_t, &k_t, &v_t, &o_t, pos_offset)
            .unwrap();

        let got = o_t.to_vec_f32().unwrap();
        let expected = cpu_attention(&q, &k, &v, b, h, h, l, l, d, pos_offset);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "attn(prefill) idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    /// CPU reference of the chained self-attn core: q_norm → transpose →
    /// RoPE → causal attention → transpose. Used by the integration test below
    /// to verify the kernels compose into a correct self-attention forward.
    #[allow(clippy::too_many_arguments)]
    fn cpu_attention_block(
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        gamma_q: &[f32],
        gamma_k: &[f32],
        cos: &[f32],
        sin: &[f32],
        b: usize,
        l: usize,
        q_h: usize,
        kv_h: usize,
        d: usize,
        rotary_dim: usize,
        rms_eps: f32,
        pos_offset: usize,
    ) -> Vec<f32> {
        // 1. RMSNorm on Q [B*L*q_h, D] and K [B*L*kv_h, D]
        let mut q = q_in.to_vec();
        let mut k = k_in.to_vec();
        for r in 0..(b * l * q_h) {
            let row = &mut q[r * d..(r + 1) * d];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let scale = (ms + rms_eps).sqrt().recip();
            for i in 0..d {
                row[i] = row[i] * scale * gamma_q[i];
            }
        }
        for r in 0..(b * l * kv_h) {
            let row = &mut k[r * d..(r + 1) * d];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let scale = (ms + rms_eps).sqrt().recip();
            for i in 0..d {
                row[i] = row[i] * scale * gamma_k[i];
            }
        }

        // 2. Transpose [B,L,H,D] → [B,H,L,D] for both Q and K (V too).
        let mut q_bhld = vec![0.0_f32; b * q_h * l * d];
        let mut k_bhld = vec![0.0_f32; b * kv_h * l * d];
        let mut v_bhld = vec![0.0_f32; b * kv_h * l * d];
        for bi in 0..b {
            for li in 0..l {
                for hi in 0..q_h {
                    for di in 0..d {
                        let src = ((bi * l + li) * q_h + hi) * d + di;
                        let dst = ((bi * q_h + hi) * l + li) * d + di;
                        q_bhld[dst] = q[src];
                    }
                }
                for hi in 0..kv_h {
                    for di in 0..d {
                        let src = ((bi * l + li) * kv_h + hi) * d + di;
                        let dst = ((bi * kv_h + hi) * l + li) * d + di;
                        k_bhld[dst] = k_in[src]; // V uses raw input, K already normed via k[]
                        v_bhld[dst] = v_in[src];
                        // NOTE: k uses normalized values
                        let kn = k[src];
                        k_bhld[dst] = kn;
                    }
                }
            }
        }

        // 3. Apply partial RoPE on Q and K.
        let half = rotary_dim / 2;
        let apply_rope = |t: &mut [f32], heads: usize| {
            for bi in 0..b {
                for hi in 0..heads {
                    for li in 0..l {
                        let base = ((bi * heads + hi) * l + li) * d;
                        for i in 0..half {
                            let c = cos[li * half + i];
                            let s = sin[li * half + i];
                            let a = t[base + i];
                            let bv = t[base + i + half];
                            t[base + i] = a * c - bv * s;
                            t[base + i + half] = a * s + bv * c;
                        }
                        // pass-through dims [rotary_dim..d] unchanged
                    }
                }
            }
        };
        apply_rope(&mut q_bhld, q_h);
        apply_rope(&mut k_bhld, kv_h);

        // 4. GQA-aware attention (BHLD layout).
        let attn = cpu_attention(&q_bhld, &k_bhld, &v_bhld, b, q_h, kv_h, l, l, d, pos_offset);

        // 5. Transpose [B,H,L,D] → [B,L,H,D].
        let mut out_blhd = vec![0.0_f32; b * l * q_h * d];
        for bi in 0..b {
            for hi in 0..q_h {
                for li in 0..l {
                    for di in 0..d {
                        let src = ((bi * q_h + hi) * l + li) * d + di;
                        let dst = ((bi * l + li) * q_h + hi) * d + di;
                        out_blhd[dst] = attn[src];
                    }
                }
            }
        }
        out_blhd
    }

    #[test]
    fn attention_block_pipeline_matches_cpu_reference() {
        // Chained self-attn core: rms_norm → transpose → rope → attention → transpose.
        // Validates that all kernels compose correctly with shape conventions
        // that match Qwen3.5-MoE (GQA, partial rotary, per-head RMSNorm).
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 1;
        let l = 4;
        let q_h = 8;
        let kv_h = 2;
        let d = 16;
        let rotary_dim = 8;
        let half = rotary_dim / 2;
        let theta = 10_000.0_f32;
        let rms_eps = 1e-6_f32;
        let pos_offset = 0;

        // Synthetic inputs (BLHD).
        let q_in: Vec<f32> = (0..b * l * q_h * d)
            .map(|i| ((i as f32) * 0.013).sin() * 1.4)
            .collect();
        let k_in: Vec<f32> = (0..b * l * kv_h * d)
            .map(|i| (((i as f32) * 0.017) + 0.5).cos() * 1.1)
            .collect();
        let v_in: Vec<f32> = (0..b * l * kv_h * d)
            .map(|i| (i as f32) * 0.005 - 0.3)
            .collect();
        let gamma_q: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let gamma_k: Vec<f32> = (0..d).map(|i| 0.95 + (i as f32) * 0.011).collect();

        // RoPE tables.
        let (cos_t, sin_t) = build_rope_tables(&ctx, rotary_dim, l, pos_offset, theta).unwrap();
        let cos_v = cos_t.to_vec_f32().unwrap();
        let sin_v = sin_t.to_vec_f32().unwrap();

        // ─── Native pipeline ───────────────────────────────────────────────
        // q, k inputs as 2D [B*L*H, D] views for rms_norm.
        let q_buf = ctx.from_slice_f32(&q_in, vec![b * l * q_h, d]).unwrap();
        let k_buf = ctx.from_slice_f32(&k_in, vec![b * l * kv_h, d]).unwrap();
        let gamma_q_t = ctx.from_slice_f32(&gamma_q, vec![d]).unwrap();
        let gamma_k_t = ctx.from_slice_f32(&gamma_k, vec![d]).unwrap();
        let q_normed_2d = ctx.zeros(vec![b * l * q_h, d], NativeDType::F32).unwrap();
        let k_normed_2d = ctx.zeros(vec![b * l * kv_h, d], NativeDType::F32).unwrap();
        lib.rms_norm(&ctx, &q_buf, &gamma_q_t, rms_eps, &q_normed_2d)
            .unwrap();
        lib.rms_norm(&ctx, &k_buf, &gamma_k_t, rms_eps, &k_normed_2d)
            .unwrap();

        // Reinterpret as [B, L, H, D] BLHD.
        let q_blhd = q_normed_2d.reshape(vec![b, l, q_h, d]).unwrap();
        let k_blhd = k_normed_2d.reshape(vec![b, l, kv_h, d]).unwrap();
        let v_blhd = ctx.from_slice_f32(&v_in, vec![b, l, kv_h, d]).unwrap();

        // Transpose to BHLD.
        let q_bhld = ctx.zeros(vec![b, q_h, l, d], NativeDType::F32).unwrap();
        let k_bhld = ctx.zeros(vec![b, kv_h, l, d], NativeDType::F32).unwrap();
        let v_bhld = ctx.zeros(vec![b, kv_h, l, d], NativeDType::F32).unwrap();
        lib.transpose_blhd(&ctx, &q_blhd, &q_bhld, 0).unwrap();
        lib.transpose_blhd(&ctx, &k_blhd, &k_bhld, 0).unwrap();
        lib.transpose_blhd(&ctx, &v_blhd, &v_bhld, 0).unwrap();

        // RoPE on Q and K (need separate output buffers — kernel forbids aliasing).
        let q_roped = ctx.zeros(vec![b, q_h, l, d], NativeDType::F32).unwrap();
        let k_roped = ctx.zeros(vec![b, kv_h, l, d], NativeDType::F32).unwrap();
        // RoPE expects [B, S, H, D] — repurpose the BHLD buffer by treating L as the
        // second axis. Our shader keys cos/sin lookup on `s = bs % seq`; with grid
        // axes (head_dim, heads, batch*seq) and seq dim = L, the BHLD layout maps
        // bs = b * H + h_idx; (bs % L) is unrelated to L. We need true [B, L, H, D]
        // ordering for RoPE indexing. Solve: transpose BHLD → BLHD, RoPE, transpose
        // back. But Q/K already came from BLHD; do RoPE directly on the BLHD buffer.
        let q_blhd_roped = ctx.zeros(vec![b, l, q_h, d], NativeDType::F32).unwrap();
        let k_blhd_roped = ctx.zeros(vec![b, l, kv_h, d], NativeDType::F32).unwrap();
        lib.rope_partial(&ctx, &q_blhd, &cos_t, &sin_t, &q_blhd_roped, half)
            .unwrap();
        lib.rope_partial(&ctx, &k_blhd, &cos_t, &sin_t, &k_blhd_roped, half)
            .unwrap();
        // Transpose roped Q/K to BHLD for attention.
        lib.transpose_blhd(&ctx, &q_blhd_roped, &q_roped, 0)
            .unwrap();
        lib.transpose_blhd(&ctx, &k_blhd_roped, &k_roped, 0)
            .unwrap();

        let attn_out = ctx.zeros(vec![b, q_h, l, d], NativeDType::F32).unwrap();
        lib.attention_causal(&ctx, &q_roped, &k_roped, &v_bhld, &attn_out, pos_offset)
            .unwrap();

        let out_blhd = ctx.zeros(vec![b, l, q_h, d], NativeDType::F32).unwrap();
        lib.transpose_blhd(&ctx, &attn_out, &out_blhd, 1).unwrap();

        let got = out_blhd.to_vec_f32().unwrap();

        // ─── CPU reference ─────────────────────────────────────────────────
        let expected = cpu_attention_block(
            &q_in, &k_in, &v_in, &gamma_q, &gamma_k, &cos_v, &sin_v, b, l, q_h, kv_h, d,
            rotary_dim, rms_eps, pos_offset,
        );

        // Compare with cosine + relative max error.
        let dot: f64 = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = expected
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = (dot / (na * nb)) as f32;
        let max_mag = got
            .iter()
            .chain(expected.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        assert!(
            cos > 0.9999,
            "block cosine {cos} (rel_max={rel}, abs_max={max_abs})"
        );
        assert!(
            rel < 5e-3,
            "block rel_max_err {rel} (cos={cos}, abs={max_abs})"
        );
    }

    /// Reference single-step SSM update mirroring the loop body in
    /// `qwen3_5_moe::linear_attn::GatedDeltaNet::forward`. Mutates `state`,
    /// returns `y`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_ssm_step(
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        b: usize,
        hv: usize,
        dv: usize,
        dk: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; b * hv * dv];
        for bi in 0..b {
            for hi in 0..hv {
                let scalar_off = bi * hv + hi;
                let beta_v = beta[scalar_off];
                let g_v = g[scalar_off];
                for di in 0..dv {
                    let state_off = ((bi * hv + hi) * dv + di) * dk;

                    // state *= g, kv_mem = sum_dk state * k.
                    let mut kv_mem = 0.0_f32;
                    for j in 0..dk {
                        let st = state[state_off + j] * g_v;
                        state[state_off + j] = st;
                        kv_mem += st * k[(bi * hv + hi) * dk + j];
                    }
                    let v_val = v[(bi * hv + hi) * dv + di];
                    let delta = (v_val - kv_mem) * beta_v;

                    // state += k * delta, y = sum_dk state * q.
                    let mut y_val = 0.0_f32;
                    for j in 0..dk {
                        let st = state[state_off + j] + k[(bi * hv + hi) * dk + j] * delta;
                        state[state_off + j] = st;
                        y_val += st * q[(bi * hv + hi) * dk + j];
                    }
                    y[(bi * hv + hi) * dv + di] = y_val;
                }
            }
        }
        y
    }

    #[test]
    fn ssm_step_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        // Small but representative shapes.
        let b = 2;
        let hv = 4;
        let dv = 8;
        let dk = 16;

        let make = |seed: u32, n: usize| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.8
                })
                .collect()
        };
        let mut state_v = make(0x10, b * hv * dv * dk);
        let q_v = make(0x20, b * hv * dk);
        let k_v = make(0x30, b * hv * dk);
        let v_v = make(0x40, b * hv * dv);
        // beta should be in (0,1) — sigmoid output; clamp range.
        let beta_v: Vec<f32> = (0..b * hv).map(|i| 0.3 + (i as f32) * 0.05).collect();
        let g_v: Vec<f32> = (0..b * hv).map(|i| 0.95 - (i as f32) * 0.01).collect();

        // ─── Native ─────────────────────────────────────────────────────────
        let state_t = ctx.from_slice_f32(&state_v, vec![b, hv, dv, dk]).unwrap();
        let q_t = ctx.from_slice_f32(&q_v, vec![b, hv, dk]).unwrap();
        let k_t = ctx.from_slice_f32(&k_v, vec![b, hv, dk]).unwrap();
        let v_t = ctx.from_slice_f32(&v_v, vec![b, hv, dv]).unwrap();
        let beta_t = ctx.from_slice_f32(&beta_v, vec![b, hv]).unwrap();
        let g_t = ctx.from_slice_f32(&g_v, vec![b, hv]).unwrap();
        let y_t = ctx.zeros(vec![b, hv, dv], NativeDType::F32).unwrap();

        lib.ssm_step(&ctx, &state_t, &q_t, &k_t, &v_t, &beta_t, &g_t, &y_t)
            .unwrap();

        let got_y = y_t.to_vec_f32().unwrap();
        let got_state = state_t.to_vec_f32().unwrap();

        // ─── CPU reference ──────────────────────────────────────────────────
        let mut state_ref = state_v.clone();
        let exp_y = cpu_ssm_step(
            &mut state_ref,
            &q_v,
            &k_v,
            &v_v,
            &beta_v,
            &g_v,
            b,
            hv,
            dv,
            dk,
        );

        for (i, (g, e)) in got_y.iter().zip(exp_y.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "ssm_step y idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
        for (i, (g, e)) in got_state.iter().zip(state_ref.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "ssm_step state idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    #[test]
    fn ssm_step_seq_loop_matches_cpu_reference() {
        // Multi-step driven from host: invoke the kernel `seq_len` times with
        // the same state buffer. Verifies state persistence + y across steps.
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let b = 1;
        let hv = 4;
        let dv = 8;
        let dk = 16;
        let seq_len = 5;

        let make = |seed: u32, n: usize| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.8
                })
                .collect()
        };
        let mut cpu_state = make(0xAB, b * hv * dv * dk);
        let q_seq = make(0x01, seq_len * b * hv * dk);
        let k_seq = make(0x02, seq_len * b * hv * dk);
        let v_seq = make(0x03, seq_len * b * hv * dv);
        let beta_seq: Vec<f32> = (0..seq_len * b * hv)
            .map(|i| 0.4 + ((i % 7) as f32) * 0.04)
            .collect();
        let g_seq: Vec<f32> = (0..seq_len * b * hv)
            .map(|i| 0.92 - ((i % 5) as f32) * 0.01)
            .collect();

        // CPU expected: sequential loop over the seq.
        let mut cpu_y_all = Vec::with_capacity(seq_len * b * hv * dv);
        for t in 0..seq_len {
            let q_t = &q_seq[t * b * hv * dk..(t + 1) * b * hv * dk];
            let k_t = &k_seq[t * b * hv * dk..(t + 1) * b * hv * dk];
            let v_t = &v_seq[t * b * hv * dv..(t + 1) * b * hv * dv];
            let beta_t = &beta_seq[t * b * hv..(t + 1) * b * hv];
            let g_t = &g_seq[t * b * hv..(t + 1) * b * hv];
            let y_t = cpu_ssm_step(&mut cpu_state, q_t, k_t, v_t, beta_t, g_t, b, hv, dv, dk);
            cpu_y_all.extend(y_t);
        }

        // Native: persistent state buffer, per-step kernel invocation.
        let mut native_state_init = make(0xAB, b * hv * dv * dk);
        let _ = native_state_init.iter_mut(); // silence unused
        let state_t = ctx
            .from_slice_f32(&make(0xAB, b * hv * dv * dk), vec![b, hv, dv, dk])
            .unwrap();
        let mut native_y_all = Vec::with_capacity(seq_len * b * hv * dv);
        for t in 0..seq_len {
            let q_t = ctx
                .from_slice_f32(
                    &q_seq[t * b * hv * dk..(t + 1) * b * hv * dk],
                    vec![b, hv, dk],
                )
                .unwrap();
            let k_t = ctx
                .from_slice_f32(
                    &k_seq[t * b * hv * dk..(t + 1) * b * hv * dk],
                    vec![b, hv, dk],
                )
                .unwrap();
            let v_t = ctx
                .from_slice_f32(
                    &v_seq[t * b * hv * dv..(t + 1) * b * hv * dv],
                    vec![b, hv, dv],
                )
                .unwrap();
            let beta_t = ctx
                .from_slice_f32(&beta_seq[t * b * hv..(t + 1) * b * hv], vec![b, hv])
                .unwrap();
            let g_t = ctx
                .from_slice_f32(&g_seq[t * b * hv..(t + 1) * b * hv], vec![b, hv])
                .unwrap();
            let y_t = ctx.zeros(vec![b, hv, dv], NativeDType::F32).unwrap();
            lib.ssm_step(&ctx, &state_t, &q_t, &k_t, &v_t, &beta_t, &g_t, &y_t)
                .unwrap();
            native_y_all.extend(y_t.to_vec_f32().unwrap());
        }

        for (i, (g, e)) in native_y_all.iter().zip(cpu_y_all.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "ssm seq-loop idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
        let final_state = state_t.to_vec_f32().unwrap();
        for (i, (g, e)) in final_state.iter().zip(cpu_state.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "ssm final state idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    #[test]
    fn rms_norm_weightless_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        for &hidden in &[64usize, 128, 1024] {
            let rows = 4;
            let x: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i as f32) * 0.013).sin() * 1.7)
                .collect();
            let eps = 1e-6_f32;

            let x_t = ctx.from_slice_f32(&x, vec![rows, hidden]).unwrap();
            let y_t = ctx.zeros(vec![rows, hidden], NativeDType::F32).unwrap();
            lib.rms_norm_weightless(&ctx, &x_t, eps, &y_t).unwrap();

            let got = y_t.to_vec_f32().unwrap();
            for r in 0..rows {
                let row = &x[r * hidden..(r + 1) * hidden];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / (hidden as f32);
                let scale = (ms + eps).sqrt().recip();
                for i in 0..hidden {
                    let expected = row[i] * scale;
                    let g = got[r * hidden + i];
                    let rel = (g - expected).abs() / expected.abs().max(1e-6);
                    assert!(
                        rel < 1e-4,
                        "weightless_rms hidden={hidden} r={r} i={i} got={g} expected={expected} rel={rel}"
                    );
                }
            }
        }
    }

    #[test]
    fn softplus_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let xs: Vec<f32> = vec![-25.0, -3.0, -0.1, 0.0, 0.1, 3.0, 25.0, 50.0];
        let x_t = ctx.from_slice_f32(&xs, vec![xs.len()]).unwrap();
        let y_t = ctx.zeros(vec![xs.len()], NativeDType::F32).unwrap();
        lib.softplus(&ctx, &x_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for (i, &x) in xs.iter().enumerate() {
            let expected = x.max(0.0) + ((-x.abs()).exp() + 1.0).ln();
            let rel = (got[i] - expected).abs() / expected.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "softplus i={i} x={x} got={} expected={expected}",
                got[i]
            );
        }
    }

    #[test]
    fn silu_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let xs: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.4).collect();
        let x_t = ctx.from_slice_f32(&xs, vec![xs.len()]).unwrap();
        let y_t = ctx.zeros(vec![xs.len()], NativeDType::F32).unwrap();
        lib.silu(&ctx, &x_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for (i, &x) in xs.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            let rel = (got[i] - expected).abs() / expected.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "silu i={i} x={x} got={} expected={expected}",
                got[i]
            );
        }
    }

    #[test]
    fn sigmoid_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let xs: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let x_t = ctx.from_slice_f32(&xs, vec![xs.len()]).unwrap();
        let y_t = ctx.zeros(vec![xs.len()], NativeDType::F32).unwrap();
        lib.sigmoid(&ctx, &x_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for (i, &x) in xs.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-x).exp());
            let rel = (got[i] - expected).abs() / expected.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "sigmoid i={i} x={x} got={} expected={expected}",
                got[i]
            );
        }
    }

    #[test]
    fn rms_norm_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        for hidden in [128usize, 1024, 2048] {
            let rows = 3;
            let x: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i as f32) * 0.013).sin() * 1.7)
                .collect();
            let gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
            let eps = 1e-6_f32;

            let x_t = ctx.from_slice_f32(&x, vec![rows, hidden]).unwrap();
            let gamma_t = ctx.from_slice_f32(&gamma, vec![hidden]).unwrap();
            let y_t = ctx.zeros(vec![rows, hidden], NativeDType::F32).unwrap();

            lib.rms_norm(&ctx, &x_t, &gamma_t, eps, &y_t).unwrap();

            let got = y_t.to_vec_f32().unwrap();
            let expected = cpu_rms_norm(&x, &gamma, hidden, eps);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let err = (g - e).abs();
                let rel = err / e.abs().max(1e-6);
                assert!(
                    rel < 1e-4,
                    "rms_norm hidden={hidden} idx={i} got={g} expected={e} rel_err={rel}"
                );
            }
        }
    }

    // ─── Linear-attn helpers (A.8-C.3) ──────────────────────────────────────

    fn lcg(seed: u32, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * scale
            })
            .collect()
    }

    #[test]
    fn broadcast_add_per_head_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let (b, s, hv) = (1, 5, 32);
        let x = lcg(0x101, b * s * hv, 1.5);
        let bias = lcg(0x102, hv, 0.7);
        let x_t = ctx.from_slice_f32(&x, vec![b, s, hv]).unwrap();
        let bias_t = ctx.from_slice_f32(&bias, vec![hv]).unwrap();
        let y_t = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        lib.broadcast_add_per_head(&ctx, &x_t, &bias_t, &y_t)
            .unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for i in 0..b * s * hv {
            let exp = x[i] + bias[i % hv];
            let err = (got[i] - exp).abs();
            assert!(err < 1e-5, "idx={i} got={} exp={} err={err}", got[i], exp);
        }
    }

    #[test]
    fn mul_broadcast_per_head_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let (b, s, hv) = (1, 7, 16);
        let x = lcg(0x201, b * s * hv, 1.0);
        let scale = lcg(0x202, hv, 1.2);
        let x_t = ctx.from_slice_f32(&x, vec![b, s, hv]).unwrap();
        let scale_t = ctx.from_slice_f32(&scale, vec![hv]).unwrap();
        let y_t = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        lib.mul_broadcast_per_head(&ctx, &x_t, &scale_t, &y_t)
            .unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for i in 0..b * s * hv {
            let exp = x[i] * scale[i % hv];
            let err = (got[i] - exp).abs();
            assert!(err < 1e-5, "idx={i} got={} exp={} err={err}", got[i], exp);
        }
    }

    #[test]
    fn neg_exp_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let n = 257;
        let x = lcg(0x301, n, 2.0);
        let x_t = ctx.from_slice_f32(&x, vec![n]).unwrap();
        let y_t = ctx.zeros(vec![n], NativeDType::F32).unwrap();
        lib.neg_exp(&ctx, &x_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for i in 0..n {
            let exp = (-x[i]).exp();
            let rel = (got[i] - exp).abs() / exp.abs().max(1e-6);
            assert!(rel < 1e-5, "idx={i} got={} exp={} rel={rel}", got[i], exp);
        }
    }

    #[test]
    fn repeat_heads_blhd_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let (b, s, hk, head_dim, repeats) = (1, 4, 8, 32, 4);
        let hv = hk * repeats;
        let x = lcg(0x401, b * s * hk * head_dim, 1.0);
        let x_t = ctx.from_slice_f32(&x, vec![b, s, hk, head_dim]).unwrap();
        let y_t = ctx
            .zeros(vec![b, s, hv, head_dim], NativeDType::F32)
            .unwrap();
        lib.repeat_heads_blhd(&ctx, &x_t, &y_t, repeats).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for bi in 0..b {
            for si in 0..s {
                for hvi in 0..hv {
                    for di in 0..head_dim {
                        let hk_src = hvi / repeats;
                        let x_i = ((bi * s + si) * hk + hk_src) * head_dim + di;
                        let y_i = ((bi * s + si) * hv + hvi) * head_dim + di;
                        let err = (got[y_i] - x[x_i]).abs();
                        assert!(
                            err < 1e-6,
                            "y[{bi},{si},{hvi},{di}] got={} exp={}",
                            got[y_i],
                            x[x_i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn affine_scalar_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let n = 513;
        let x = lcg(0x501, n, 1.5);
        let scale = 0.125_f32;
        let bias = -0.375_f32;
        let x_t = ctx.from_slice_f32(&x, vec![n]).unwrap();
        let y_t = ctx.zeros(vec![n], NativeDType::F32).unwrap();
        lib.affine_scalar(&ctx, &x_t, scale, bias, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        for i in 0..n {
            let exp = x[i] * scale + bias;
            let err = (got[i] - exp).abs();
            assert!(err < 1e-6, "idx={i} got={} exp={} err={err}", got[i], exp);
        }
    }

    /// CPU reference for the depthwise causal conv1d + SiLU kernel. The CPU
    /// path mirrors the Metal kernel exactly: per-channel `kernel_size`-wide
    /// reduction over the [conv_state ++ qkv_flat] window, followed by SiLU.
    #[test]
    fn depthwise_conv1d_silu_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        // Production shape (Qwen3.5-VL-MoE): qkv_dim=8192, kernel=4, B=1.
        // Use smaller dims to keep the parity loop cheap but exercise both
        // tails (channels not multiple of typical thread group widths).
        let (b, s, k, c) = (2usize, 5usize, 4usize, 33usize);
        let x_total = k - 1 + s;
        let x_host = lcg(0xC01D, b * x_total * c, 0.7);
        let w_host = lcg(0xC02D, c * k, 0.4);

        let x_t = ctx.from_slice_f32(&x_host, vec![b, x_total, c]).unwrap();
        let w_t = ctx.from_slice_f32(&w_host, vec![c, k]).unwrap();
        let y_t = ctx.zeros(vec![b, s, c], NativeDType::F32).unwrap();

        lib.depthwise_conv1d_silu(&ctx, &x_t, &w_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();

        for bi in 0..b {
            for t in 0..s {
                for ci in 0..c {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        let xv = x_host[bi * x_total * c + (t + ki) * c + ci];
                        let wv = w_host[ci * k + ki];
                        acc += xv * wv;
                    }
                    let sig = 1.0f32 / (1.0f32 + (-acc).exp());
                    let expect = acc * sig;
                    let g = got[bi * s * c + t * c + ci];
                    assert!(
                        (g - expect).abs() < 1e-4,
                        "b={bi} t={t} c={ci} got={g} exp={expect}"
                    );
                }
            }
        }
    }

    #[test]
    fn encode_variants_match_committing_versions() {
        // Sanity test: dispatch all five new kernels through their `encode_*`
        // variants in a single fused command buffer and verify the outputs
        // match the per-kernel committed variants. Catches encoder lifecycle
        // regressions when the wrappers are reused inside `forward_linear_attn`.
        use lumen_metal::metal::ComputeCommandEncoderRef as _Enc;
        let _ = std::any::type_name::<_Enc>(); // ensures the import is used
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let (b, s, hk, head_dim, repeats) = (1, 3, 4, 16, 2);
        let hv = hk * repeats;
        let n_bsh = b * s * hv;
        let x_bsh = lcg(0xE01, n_bsh, 1.0);
        let bias = lcg(0xE02, hv, 0.5);
        let x_blhd = lcg(0xE03, b * s * hk * head_dim, 0.8);

        // committed reference outputs
        let bs_x = ctx.from_slice_f32(&x_bsh, vec![b, s, hv]).unwrap();
        let bias_t = ctx.from_slice_f32(&bias, vec![hv]).unwrap();
        let y1_ref = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y2_ref = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y3_ref = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y4_ref = ctx
            .zeros(vec![b, s, hv, head_dim], NativeDType::F32)
            .unwrap();
        let y5_ref = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let blhd_x = ctx
            .from_slice_f32(&x_blhd, vec![b, s, hk, head_dim])
            .unwrap();
        lib.broadcast_add_per_head(&ctx, &bs_x, &bias_t, &y1_ref)
            .unwrap();
        lib.mul_broadcast_per_head(&ctx, &y1_ref, &bias_t, &y2_ref)
            .unwrap();
        lib.neg_exp(&ctx, &y2_ref, &y3_ref).unwrap();
        lib.repeat_heads_blhd(&ctx, &blhd_x, &y4_ref, repeats)
            .unwrap();
        lib.affine_scalar(&ctx, &y3_ref, 0.5, 0.25, &y5_ref)
            .unwrap();

        // fused-encoder run
        let y1 = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y2 = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y3 = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let y4 = ctx
            .zeros(vec![b, s, hv, head_dim], NativeDType::F32)
            .unwrap();
        let y5 = ctx.zeros(vec![b, s, hv], NativeDType::F32).unwrap();
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        lib.encode_broadcast_add_per_head(&enc, &bs_x, &bias_t, &y1)
            .unwrap();
        lib.encode_mul_broadcast_per_head(&enc, &y1, &bias_t, &y2)
            .unwrap();
        lib.encode_neg_exp(&enc, &y2, &y3).unwrap();
        lib.encode_repeat_heads_blhd(&enc, &blhd_x, &y4, repeats)
            .unwrap();
        lib.encode_affine_scalar(&enc, &y3, 0.5, 0.25, &y5).unwrap();
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        for (label, a, e) in [
            ("y1", y1.to_vec_f32().unwrap(), y1_ref.to_vec_f32().unwrap()),
            ("y2", y2.to_vec_f32().unwrap(), y2_ref.to_vec_f32().unwrap()),
            ("y3", y3.to_vec_f32().unwrap(), y3_ref.to_vec_f32().unwrap()),
            ("y4", y4.to_vec_f32().unwrap(), y4_ref.to_vec_f32().unwrap()),
            ("y5", y5.to_vec_f32().unwrap(), y5_ref.to_vec_f32().unwrap()),
        ] {
            for (i, (g, x)) in a.iter().zip(e.iter()).enumerate() {
                let err = (g - x).abs();
                assert!(err < 1e-5, "{label} idx={i} got={g} exp={x} err={err}");
            }
        }
    }
}
