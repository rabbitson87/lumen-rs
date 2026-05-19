//! GPU sampling primitives for the Qwen3.5-VL-MoE forward path (Phase A.6).
//!
//! Currently exposes a single argmax kernel — used by `generate_with_opts`'s
//! greedy fast path (temperature == 0, top_k == 0, repeat_penalty == 1.0) to
//! avoid pulling the full 248K-vocab logits vector through host memory.
//!
//! Future expansion: top-k partial sort, top-p cumulative mask, multinomial.
//! All as additions to [`SamplingKernels`].

use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use crate::metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger};
use anyhow::{Result, anyhow};

const SHADER_SRC: &str = include_str!("shaders/sampling.metal");

/// Compile-time cap matching the kernel's `K_MAX` constant. Bumping requires
/// matching changes in `shaders/sampling.metal`.
pub const TOPK_K_MAX: u32 = 32;

/// Lazily-compiled sampling pipelines. One instance per process is enough — it
/// holds only `ComputePipelineState`s, which are cheap to clone.
pub struct SamplingKernels {
    argmax_f32: ComputePipelineState,
    apply_token_penalties_f32: ComputePipelineState,
    /// fused top-k + top-p + Gumbel-max sampler. Used by the
    /// stochastic decode path when the caller wants nucleus / top-k filtering
    /// without paying a 248K-vocab host transfer.
    topk_topp_gumbel: ComputePipelineState,
}

impl SamplingKernels {
    /// Compile the sampling shader and pull out every kernel by name.
    pub fn new(device: &Device) -> Result<Self> {
        let opts = crate::metal::new_compile_options();
        opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_0);
        let lib = device
            .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
            .map_err(|e| anyhow!("sampling shader compile: {e}"))?;
        let argmax = lib
            .get_function("argmax_f32", None)
            .map_err(|e| anyhow!("sampling: get_function argmax_f32: {e}"))?;
        let argmax_f32 = device
            .new_compute_pipeline_state_with_function(&argmax)
            .map_err(|e| anyhow!("sampling: pipeline argmax_f32: {e}"))?;
        let penalties = lib
            .get_function("apply_token_penalties_f32", None)
            .map_err(|e| anyhow!("sampling: get_function apply_token_penalties_f32: {e}"))?;
        let apply_token_penalties_f32 = device
            .new_compute_pipeline_state_with_function(&penalties)
            .map_err(|e| anyhow!("sampling: pipeline apply_token_penalties_f32: {e}"))?;
        let topk_fn = lib
            .get_function("topk_topp_gumbel_argmax_f32", None)
            .map_err(|e| anyhow!("sampling: get_function topk_topp_gumbel_argmax_f32: {e}"))?;
        let topk_topp_gumbel = device
            .new_compute_pipeline_state_with_function(&topk_fn)
            .map_err(|e| anyhow!("sampling: pipeline topk_topp_gumbel_argmax_f32: {e}"))?;
        Ok(Self {
            argmax_f32,
            apply_token_penalties_f32,
            topk_topp_gumbel,
        })
    }

    /// Phase B+ (2026-04-27) GPU penalty application using caller-provided
    /// pre-allocated scratch buffers. Avoids the per-call Metal allocator
    /// round-trip that the original `apply_token_penalties_f32` paid; caller
    /// is expected to hold long-lived `idx_buf`/`mul_buf` (StorageModeShared)
    /// of capacity ≥ `n_pairs`.
    ///
    /// Memcpys the host-side `token_idx_data` and `multipliers_data` slices
    /// into the buffers' `contents()` pointers and dispatches the kernel.
    /// All other contracts match `apply_token_penalties_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_token_penalties_f32_into(
        &self,
        queue: &crate::metal::CommandQueueRef,
        logits_buf: &Buffer,
        logits_offset: u64,
        idx_buf: &Buffer,
        mul_buf: &Buffer,
        idx_capacity: usize,
        token_idx_data: &[u32],
        multipliers_data: &[f32],
    ) -> Result<()> {
        if token_idx_data.is_empty() {
            return Ok(());
        }
        if token_idx_data.len() != multipliers_data.len() {
            return Err(anyhow!(
                "apply_token_penalties_f32_into: idx len {} != mul len {}",
                token_idx_data.len(),
                multipliers_data.len()
            ));
        }
        if token_idx_data.len() > idx_capacity {
            return Err(anyhow!(
                "apply_token_penalties_f32_into: input {} exceeds scratch capacity {}",
                token_idx_data.len(),
                idx_capacity
            ));
        }
        let n = token_idx_data.len() as u32;
        // SAFETY: the scratch buffers are `StorageModeShared`, fixed-size,
        // owned by the caller for the duration of this call. We bound the
        // copy by `idx_capacity * 4` bytes (validated above) and the source
        // slice has the right element count.
        unsafe {
            let dst = idx_buf.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(token_idx_data.as_ptr(), dst, token_idx_data.len());
            let dst = mul_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(multipliers_data.as_ptr(), dst, multipliers_data.len());
        }

        let enc = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        enc.set_label("lumen:apply_token_penalties");
        enc.set_compute_pipeline_state(&self.apply_token_penalties_f32);
        enc.set_buffer(0, Some(logits_buf), logits_offset as usize);
        enc.set_buffer(1, Some(idx_buf), 0);
        enc.set_buffer(2, Some(mul_buf), 0);
        enc.set_bytes_directly(3, 4, &n as *const _ as *const _);

        let max_threads = self
            .apply_token_penalties_f32
            .max_total_threads_per_threadgroup() as u64;
        let tg_size = (n as u64).min(max_threads).max(1);
        enc.dispatch_threads(
            MTLSize {
                width: n as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size as usize,
                height: 1,
                depth: 1,
            },
        );
        drop(enc);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// penalty application. Modifies
    /// `logits` in place at the indices supplied by `token_idx`, applying the
    /// same sign-aware multiplier the CPU sampler used.
    ///
    /// Caller responsibilities:
    ///   - Pre-aggregate so each `token_idx` is unique (combine repeat-
    ///     penalty^count and ngram-penalty^matches into a single multiplier
    ///     per index). Without uniqueness, parallel writes from different
    ///     threads to the same logit position would race.
    ///   - `token_idx_data` and `multipliers_data` must have identical length.
    ///   - Empty input is a fast no-op (no kernel launch, returns Ok).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_token_penalties_f32(
        &self,
        queue: &crate::metal::CommandQueueRef,
        device: &Device,
        logits_buf: &Buffer,
        logits_offset: u64,
        token_idx_data: &[u32],
        multipliers_data: &[f32],
    ) -> Result<()> {
        if token_idx_data.is_empty() {
            return Ok(());
        }
        if token_idx_data.len() != multipliers_data.len() {
            return Err(anyhow!(
                "apply_token_penalties_f32: idx len {} != mul len {}",
                token_idx_data.len(),
                multipliers_data.len()
            ));
        }
        let n = token_idx_data.len() as u32;
        let idx_bytes = (token_idx_data.len() * 4) as NSUInteger;
        let mul_bytes = (multipliers_data.len() * 4) as NSUInteger;
        let idx_buf = device
            .new_buffer_with_data(
                token_idx_data.as_ptr() as *const _,
                idx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow!("apply_token_penalties_f32: idx buffer alloc: {e}"))?;
        let mul_buf = device
            .new_buffer_with_data(
                multipliers_data.as_ptr() as *const _,
                mul_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow!("apply_token_penalties_f32: mul buffer alloc: {e}"))?;

        let enc = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        enc.set_label("lumen:apply_token_penalties");
        enc.set_compute_pipeline_state(&self.apply_token_penalties_f32);
        enc.set_buffer(0, Some(logits_buf), logits_offset as usize);
        enc.set_buffer(1, Some(&idx_buf), 0);
        enc.set_buffer(2, Some(&mul_buf), 0);
        enc.set_bytes_directly(3, 4, &n as *const _ as *const _);

        let max_threads = self
            .apply_token_penalties_f32
            .max_total_threads_per_threadgroup() as u64;
        let tg_size = (n as u64).min(max_threads).max(1);
        // 1D dispatch with `dispatch_threads` so out-of-range threads bail.
        enc.dispatch_threads(
            MTLSize {
                width: n as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size as usize,
                height: 1,
                depth: 1,
            },
        );
        drop(enc);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// A.1 (greedy fast path) — fuse `apply_token_penalties_f32` + `argmax_f32`
    /// into one command buffer with a single commit + wait. Eliminates the
    /// commit/wait round-trip that the two-call sequence pays.
    ///
    /// Behavior is identical to calling `apply_token_penalties_f32_into`
    /// followed by `argmax_f32` — Metal's intra-cmd-buffer hazard tracking
    /// inserts the necessary memory barrier between the two encoders so the
    /// argmax reads the penalty-applied logits.
    ///
    /// Empty penalty list short-circuits the penalty encoder (no kernel
    /// launch) and runs argmax alone, still in one cmd buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_penalties_then_argmax(
        &self,
        queue: &crate::metal::CommandQueueRef,
        logits_buf: &Buffer,
        logits_offset: u64,
        n_logits: u32,
        idx_buf: &Buffer,
        mul_buf: &Buffer,
        idx_capacity: usize,
        token_idx_data: &[u32],
        multipliers_data: &[f32],
        out_buf: &Buffer,
        out_offset: u64,
    ) -> Result<()> {
        if n_logits == 0 {
            return Err(anyhow!("apply_penalties_then_argmax: n_logits must be > 0"));
        }
        if token_idx_data.len() != multipliers_data.len() {
            return Err(anyhow!(
                "apply_penalties_then_argmax: idx len {} != mul len {}",
                token_idx_data.len(),
                multipliers_data.len()
            ));
        }
        if token_idx_data.len() > idx_capacity {
            return Err(anyhow!(
                "apply_penalties_then_argmax: input {} exceeds scratch capacity {}",
                token_idx_data.len(),
                idx_capacity
            ));
        }

        // Stage host data into the caller-owned shared scratch buffers (no
        // per-call allocator round-trip). Skipped when the list is empty.
        let n_pen = token_idx_data.len() as u32;
        if n_pen > 0 {
            // SAFETY: scratch buffers are StorageModeShared with capacity
            // `idx_capacity`; we copy ≤ `idx_capacity` u32 / f32 elements.
            unsafe {
                let dst = idx_buf.contents() as *mut u32;
                std::ptr::copy_nonoverlapping(token_idx_data.as_ptr(), dst, token_idx_data.len());
                let dst = mul_buf.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(
                    multipliers_data.as_ptr(),
                    dst,
                    multipliers_data.len(),
                );
            }
        }

        // Encoder 1 — penalty (skip when no pairs).
        // candle_metal_kernels::ComputeCommandEncoder has a Drop impl that
        // calls end_encoding(), so we MUST NOT call it explicitly (B4 found
        // that the double-end raises an Apple runtime assertion). Each
        // encoder is scoped so it drops before the next one is created.
        if n_pen > 0 {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            enc.set_label("lumen:apply_token_penalties");
            enc.set_compute_pipeline_state(&self.apply_token_penalties_f32);
            enc.set_buffer(0, Some(logits_buf), logits_offset as usize);
            enc.set_buffer(1, Some(idx_buf), 0);
            enc.set_buffer(2, Some(mul_buf), 0);
            enc.set_bytes_directly(3, 4, &n_pen as *const _ as *const _);
            let max_threads = self
                .apply_token_penalties_f32
                .max_total_threads_per_threadgroup() as u64;
            let tg_size = (n_pen as u64).min(max_threads).max(1);
            enc.dispatch_threads(
                MTLSize {
                    width: n_pen as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size as usize,
                    height: 1,
                    depth: 1,
                },
            );
            // enc drops at end of `if` block → Drop ends the encoder.
        }

        // Encoder 2 — argmax over the (now penalty-applied) logits. Metal
        // inserts the implicit hazard barrier between encoders writing/reading
        // the same buffer, so no explicit fence is needed.
        let max_threads = self.argmax_f32.max_total_threads_per_threadgroup() as u64;
        let mut tg_size: u64 = 1;
        let want = (n_logits as u64).min(max_threads).min(1024);
        while tg_size * 2 <= want {
            tg_size *= 2;
        }
        {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            enc.set_label("lumen:argmax");
            enc.set_compute_pipeline_state(&self.argmax_f32);
            enc.set_buffer(0, Some(logits_buf), logits_offset as usize);
            enc.set_buffer(1, Some(out_buf), out_offset as usize);
            enc.set_bytes_directly(2, 4, &n_logits as *const _ as *const _);
            enc.set_threadgroup_memory_length(0, (tg_size * 4) as usize);
            enc.set_threadgroup_memory_length(1, (tg_size * 4) as usize);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size as usize,
                    height: 1,
                    depth: 1,
                },
            );
            // enc drops at end of inner block → Drop ends the encoder before commit.
        }

        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Cooperative argmax over `[n]` f32 starting at `x_offset` bytes inside
    /// `x_buf`. Writes a single u32 index into `out_buf` at `out_offset`.
    ///
    /// The dispatch uses a single threadgroup so the tree reduction in shared
    /// memory finishes in one pass — fine for vocab-size inputs (248K f32
    /// fits with the 32-bit per-thread reduction step), avoids a second
    /// kernel launch.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_f32(
        &self,
        queue: &crate::metal::CommandQueueRef,
        x_buf: &Buffer,
        x_offset: u64,
        n: u32,
        out_buf: &Buffer,
        out_offset: u64,
    ) -> Result<()> {
        if n == 0 {
            return Err(anyhow!("argmax_f32: n must be > 0"));
        }
        let max_threads = self.argmax_f32.max_total_threads_per_threadgroup() as u64;
        // Pick a power-of-two tg_size up to min(max_threads, 1024). 256/512 is
        // more than enough — each thread already strides through n items.
        let mut tg_size: u64 = 1;
        let want = (n as u64).min(max_threads).min(1024);
        while tg_size * 2 <= want {
            tg_size *= 2;
        }

        let enc = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        enc.set_label("lumen:argmax");
        enc.set_compute_pipeline_state(&self.argmax_f32);
        enc.set_buffer(0, Some(x_buf), x_offset as usize);
        enc.set_buffer(1, Some(out_buf), out_offset as usize);
        enc.set_bytes_directly(2, 4, &n as *const _ as *const _);
        // shared_v: tg_size f32 (4 bytes each), shared_i: tg_size u32 (4 bytes each).
        enc.set_threadgroup_memory_length(0, (tg_size * 4) as usize);
        enc.set_threadgroup_memory_length(1, (tg_size * 4) as usize);
        enc.dispatch_thread_groups(
            MTLSize {
                width: (1) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
            MTLSize {
                width: (tg_size) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
        );
        drop(enc);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Fused top-k + top-p + Gumbel-max GPU sample.
    ///
    /// Single-threadgroup dispatch (64 threads). Reads `n_logits` floats
    /// starting at `logits_off` bytes inside `logits_buf` (assumed to be
    /// already penalty-applied), produces a per-thread top-`top_k`
    /// sorted-descending list, tree-reduces in threadgroup memory, then on
    /// thread 0 runs softmax + top-p truncation + Gumbel-max sampling and
    /// writes the resulting token index (u32) into `out_buf` at `out_off`.
    ///
    /// Validation:
    ///   - `n_logits > 0`
    ///   - `1 <= top_k <= TOPK_K_MAX (32)` (caller must clamp / pre-validate)
    ///   - `0.0 < top_p <= 1.0`
    ///   - `inv_temp > 0`
    ///
    /// Threadgroup memory budget: 64 * 32 * (4 + 4) = 16 KB (well under the
    /// 32 KB per-threadgroup limit on M3 Max).
    #[allow(clippy::too_many_arguments)]
    pub fn topk_topp_gumbel_argmax_f32_into(
        &self,
        queue: &crate::metal::CommandQueueRef,
        logits_buf: &Buffer,
        logits_off: u64,
        n_logits: u32,
        top_k: u32,
        top_p: f32,
        inv_temp: f32,
        seed_lo: u32,
        seed_hi: u32,
        out_buf: &Buffer,
        out_off: u64,
    ) -> Result<()> {
        if n_logits == 0 {
            return Err(anyhow!(
                "topk_topp_gumbel_argmax_f32_into: n_logits must be > 0"
            ));
        }
        if top_k == 0 || top_k > TOPK_K_MAX {
            return Err(anyhow!(
                "topk_topp_gumbel_argmax_f32_into: top_k {top_k} out of range [1, {TOPK_K_MAX}]"
            ));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(anyhow!(
                "topk_topp_gumbel_argmax_f32_into: top_p {top_p} out of range (0, 1]"
            ));
        }
        if !(inv_temp > 0.0) {
            return Err(anyhow!(
                "topk_topp_gumbel_argmax_f32_into: inv_temp {inv_temp} must be > 0"
            ));
        }

        // Match kernel design: 1 threadgroup, 64 threads. K_MAX is a
        // compile-time cap in the shader (constant uint K_MAX = 32). The
        // threadgroup memory layout is [tg_size × K_MAX] for both vals and
        // idxs, so 64 × 32 × 4 = 8192 bytes per array.
        let tg_size: u64 = 64;
        let k_max: u64 = TOPK_K_MAX as u64;
        let tg_mem_bytes = (tg_size * k_max * 4) as usize; // 8 KiB per array

        {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            enc.set_label("lumen:topk_topp_gumbel");
            enc.set_compute_pipeline_state(&self.topk_topp_gumbel);
            enc.set_buffer(0, Some(logits_buf), logits_off as usize);
            enc.set_buffer(1, Some(out_buf), out_off as usize);
            enc.set_bytes_directly(2, 4, &n_logits as *const _ as *const _);
            enc.set_bytes_directly(3, 4, &top_k as *const _ as *const _);
            enc.set_bytes_directly(4, 4, &top_p as *const _ as *const _);
            enc.set_bytes_directly(5, 4, &inv_temp as *const _ as *const _);
            enc.set_bytes_directly(6, 4, &seed_lo as *const _ as *const _);
            enc.set_bytes_directly(7, 4, &seed_hi as *const _ as *const _);
            enc.set_threadgroup_memory_length(0, tg_mem_bytes);
            enc.set_threadgroup_memory_length(1, tg_mem_bytes);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size as usize,
                    height: 1,
                    depth: 1,
                },
            );
            // enc drops here → ComputeCommandEncoder Drop calls end_encoding().
        }
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Allocate a 1-element u32 buffer (for `argmax_f32`'s output). Convenience.
    pub fn alloc_u32_scalar(device: &Device) -> Buffer {
        device
            .new_buffer(4, MTLResourceOptions::StorageModeShared)
            .expect("alloc_u32_scalar: Metal buffer alloc failed")
    }

    /// Read the single u32 stored at the start of `buf`.
    pub fn read_u32_scalar(buf: &Buffer) -> u32 {
        let ptr = buf.contents() as *const u32;
        unsafe { ptr.read_volatile() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Device;

    fn upload_f32(device: &Device, data: &[f32]) -> Buffer {
        let bytes = (data.len() * 4) as NSUInteger;
        device
            .new_buffer_with_data(
                data.as_ptr() as *const _,
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("upload_f32: Metal buffer alloc failed")
    }

    #[test]
    fn argmax_matches_cpu_on_small_vector() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        let v: Vec<f32> = vec![0.1, 0.5, -1.0, 0.7, 0.6, 0.7001, 0.2];
        let cpu_idx = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;

        let x_buf = upload_f32(&device, &v);
        let out_buf = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .argmax_f32(&queue, &x_buf, 0, v.len() as u32, &out_buf, 0)
            .unwrap();
        let gpu_idx = SamplingKernels::read_u32_scalar(&out_buf);
        assert_eq!(gpu_idx, cpu_idx, "argmax mismatch on small vector");
    }

    #[test]
    fn argmax_matches_cpu_on_vocab_size() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        // 248K vocab, deterministic LCG-style fill, with one inserted spike.
        let n = 248_320usize;
        let mut v = Vec::with_capacity(n);
        let mut s = 0xDEAD_BEEFu32;
        for _ in 0..n {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            v.push(((s >> 8 & 0xFFFF) as f32 / 65535.0 - 0.5) * 10.0);
        }
        let spike_idx = 200_000;
        v[spike_idx] = 1e9;

        let cpu_idx = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        assert_eq!(cpu_idx, spike_idx as u32);

        let x_buf = upload_f32(&device, &v);
        let out_buf = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .argmax_f32(&queue, &x_buf, 0, n as u32, &out_buf, 0)
            .unwrap();
        let gpu_idx = SamplingKernels::read_u32_scalar(&out_buf);
        assert_eq!(gpu_idx, cpu_idx, "argmax mismatch on vocab-sized vector");
    }

    #[test]
    fn argmax_respects_offset() {
        // Place data at byte offset 16 to verify x_offset parameter wires through.
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        // Buffer of 8 floats; argmax over the trailing 4 starting at offset 16.
        let v: Vec<f32> = vec![99.0, 99.0, 99.0, 99.0, 0.1, 0.7, 0.3, 0.5];
        // Trailing 4 = [0.1, 0.7, 0.3, 0.5] → argmax index 1 inside that window.
        let x_buf = upload_f32(&device, &v);
        let out_buf = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .argmax_f32(&queue, &x_buf, 16, 4, &out_buf, 0)
            .unwrap();
        let gpu_idx = SamplingKernels::read_u32_scalar(&out_buf);
        assert_eq!(gpu_idx, 1);
    }

    fn alloc_shared_u32(device: &Device, capacity: usize) -> Buffer {
        device
            .new_buffer(
                (capacity * 4) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("alloc shared u32 buffer")
    }

    fn alloc_shared_f32(device: &Device, capacity: usize) -> Buffer {
        device
            .new_buffer(
                (capacity * 4) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("alloc shared f32 buffer")
    }

    #[test]
    fn fused_penalty_then_argmax_matches_two_call_sequence() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        // Logits with a clear pre-penalty winner that gets penalized below
        // a runner-up — proves the argmax sees the penalty-applied buffer.
        // pre:    [0.10, 0.50, 0.30, 1.00, 0.20]  argmax=3 (value 1.0)
        // penalty[3]=4.0 → 1.0/4.0 = 0.25 → argmax should now be index 1 (0.5)
        let logits_initial: Vec<f32> = vec![0.10, 0.50, 0.30, 1.00, 0.20];
        let n = logits_initial.len();

        // Two-call reference: same kernels, separate cmd buffers.
        let ref_logits = upload_f32(&device, &logits_initial);
        let ref_out = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .apply_token_penalties_f32(&queue, &device, &ref_logits, 0, &[3u32], &[4.0f32])
            .unwrap();
        kernels
            .argmax_f32(&queue, &ref_logits, 0, n as u32, &ref_out, 0)
            .unwrap();
        let ref_idx_val = SamplingKernels::read_u32_scalar(&ref_out);
        assert_eq!(ref_idx_val, 1, "two-call reference should return idx 1");

        // Fused-call path: same input, scratch buffers + single cmd buffer.
        let fused_logits = upload_f32(&device, &logits_initial);
        let scratch_idx = alloc_shared_u32(&device, 16);
        let scratch_mul = alloc_shared_f32(&device, 16);
        let fused_out = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .apply_penalties_then_argmax(
                &queue,
                &fused_logits,
                0,
                n as u32,
                &scratch_idx,
                &scratch_mul,
                16,
                &[3u32],
                &[4.0f32],
                &fused_out,
                0,
            )
            .unwrap();
        let fused_idx_val = SamplingKernels::read_u32_scalar(&fused_out);
        assert_eq!(fused_idx_val, ref_idx_val);

        // Logit buffer state must also match (penalty applied identically).
        let ref_after: &[f32] =
            unsafe { std::slice::from_raw_parts(ref_logits.contents() as *const f32, n) };
        let fused_after: &[f32] =
            unsafe { std::slice::from_raw_parts(fused_logits.contents() as *const f32, n) };
        for i in 0..n {
            assert!(
                (ref_after[i] - fused_after[i]).abs() < 1e-6,
                "post-penalty logits diverge at idx {i}: ref={} fused={}",
                ref_after[i],
                fused_after[i],
            );
        }
    }

    #[test]
    fn fused_with_empty_penalty_runs_argmax_only() {
        // Empty penalty list must still produce argmax over the original logits.
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        let logits: Vec<f32> = vec![0.1, 0.7, 0.2, 0.5];
        let logits_buf = upload_f32(&device, &logits);
        let scratch_idx = alloc_shared_u32(&device, 4);
        let scratch_mul = alloc_shared_f32(&device, 4);
        let out_buf = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .apply_penalties_then_argmax(
                &queue,
                &logits_buf,
                0,
                logits.len() as u32,
                &scratch_idx,
                &scratch_mul,
                4,
                &[],
                &[],
                &out_buf,
                0,
            )
            .unwrap();
        let idx = SamplingKernels::read_u32_scalar(&out_buf);
        assert_eq!(idx, 1, "argmax over [0.1, 0.7, 0.2, 0.5] should be 1");
    }

    #[test]
    fn fused_on_vocab_size_input() {
        // 248K vocab + sparse penalty list that flips the winner.
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        let kernels = SamplingKernels::new(&device).unwrap();

        let n = 248_320usize;
        let mut v = vec![0.0f32; n];
        // Top-1 at idx 100, second-best at idx 50.
        v[100] = 10.0;
        v[50] = 5.0;
        // Penalty index 100 with multiplier 5.0 → 10.0 / 5.0 = 2.0, so idx 50 wins.
        let logits_buf = upload_f32(&device, &v);
        let scratch_idx = alloc_shared_u32(&device, 16);
        let scratch_mul = alloc_shared_f32(&device, 16);
        let out_buf = SamplingKernels::alloc_u32_scalar(&device);
        kernels
            .apply_penalties_then_argmax(
                &queue,
                &logits_buf,
                0,
                n as u32,
                &scratch_idx,
                &scratch_mul,
                16,
                &[100u32],
                &[5.0f32],
                &out_buf,
                0,
            )
            .unwrap();
        let idx = SamplingKernels::read_u32_scalar(&out_buf);
        assert_eq!(idx, 50, "penalty should have flipped argmax from 100 to 50");
    }
}
