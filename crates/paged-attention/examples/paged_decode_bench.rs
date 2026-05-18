//! Real-scale correctness & performance test for PagedAttention kernels.
//!
//! Simulates Gemma 4 31B (or E4B) attention dimensions with randomly
//! generated K/V tensors that match the shape produced by real projections.
//! Compares GPU paged_attention_decode output against a CPU reference
//! implementation of the same attention math.
//!
//! Usage:
//!   cargo run --release --example paged_decode_bench -p paged-attention
//!   MODEL=31B cargo run --release --example paged_decode_bench -p paged-attention
//!   MODEL=E4B cargo run --release --example paged_decode_bench -p paged-attention
//!   CTX_LEN=2048 cargo run --release --example paged_decode_bench -p paged-attention

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_metal_kernels::metal::{
    Buffer, CommandBuffer, CommandQueue, CommandSemaphore, Device, MTLResourceOptions,
    create_command_buffer,
};
use paged_attention::block_allocator::{BlockAllocator, BlockConfig, LayerAttentionConfig};
use paged_attention::block_table::PageTable;
use paged_attention::metal::dispatch::PagedAttentionPipelines;

/// Preset dimensions for Gemma 4 models.
struct ModelPreset {
    name: &'static str,
    n_layers: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim_sliding: u32,
    head_dim_global: u32,
    /// `sliding_window_pattern`: global layer every N layers.
    global_every: u32,
}

const E4B: ModelPreset = ModelPreset {
    name: "E4B",
    n_layers: 26,
    n_q_heads: 8,
    n_kv_heads: 2,
    head_dim_sliding: 256,
    head_dim_global: 256,
    global_every: 6,
};

const G31B: ModelPreset = ModelPreset {
    name: "31B",
    n_layers: 48,
    n_q_heads: 16,
    n_kv_heads: 8,
    head_dim_sliding: 256,
    head_dim_global: 512,
    global_every: 6,
};

fn build_config(preset: &ModelPreset, block_size: u32) -> BlockConfig {
    let layers: Vec<LayerAttentionConfig> = (0..preset.n_layers)
        .map(|i| {
            let is_sliding = (i + 1) % preset.global_every != 0;
            LayerAttentionConfig {
                n_kv_heads: preset.n_kv_heads,
                head_dim: if is_sliding {
                    preset.head_dim_sliding
                } else {
                    preset.head_dim_global
                },
                is_sliding,
            }
        })
        .collect();
    BlockConfig::from_layers(block_size, layers)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn f16_bits_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// Deterministic pseudo-random f32 in roughly [-1, 1].
fn rand_f32(seed: u64) -> f32 {
    // xorshift64
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // Map to [-1, 1]
    ((s as i32) as f32) / (i32::MAX as f32)
}

/// Generate tensor data with reproducible values.
fn gen_tensor(n_elems: usize, seed_base: u64, scale: f32) -> Vec<f32> {
    (0..n_elems)
        .map(|i| rand_f32(seed_base.wrapping_add(i as u64)) * scale)
        .collect()
}

/// Buffer helpers.
fn buffer_f16(device: &Device, data: &[f32]) -> Buffer {
    let bits: Vec<u16> = data.iter().map(|&x| f32_to_f16_bits(x)).collect();
    device
        .new_buffer_with_data(
            bits.as_ptr() as *const _,
            bits.len() * 2,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("f16 buffer alloc")
}

fn buffer_f32(device: &Device, data: &[f32]) -> Buffer {
    device
        .new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("f32 buffer alloc")
}

fn buffer_i32(device: &Device, data: &[i32]) -> Buffer {
    device
        .new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("i32 buffer alloc")
}

fn read_f32(buf: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    slice.to_vec()
}

fn new_cmd(queue: &CommandQueue) -> CommandBuffer {
    create_command_buffer(queue, Arc::new(CommandSemaphore::new())).expect("cmd buffer")
}

/// CPU reference SDPA for one head.
///
/// q: [head_dim]
/// k, v: [ctx_len, head_dim]
/// Returns: [head_dim]
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ctx_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    // Compute scores
    let mut scores = Vec::with_capacity(ctx_len);
    for t in 0..ctx_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[t * head_dim + d];
        }
        scores.push(dot * scale);
    }
    // Softmax
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    // Weighted V sum
    let mut out = vec![0.0f32; head_dim];
    for t in 0..ctx_len {
        for d in 0..head_dim {
            out[d] += probs[t] * v[t * head_dim + d];
        }
    }
    out
}

/// Run one layer's paged decode + reference + compare.
#[allow(clippy::too_many_arguments)]
fn test_layer(
    device: &Device,
    queue: &CommandQueue,
    pipelines: &PagedAttentionPipelines,
    alloc: &mut BlockAllocator,
    layer: usize,
    preset: &ModelPreset,
    ctx_len: usize,
    block_size: u32,
    head_dim: u32,
    seed: u64,
) -> Result<(f32, f32, f64, f64)> {
    // Allocate enough blocks for ctx_len tokens in this sequence
    let mut pt = PageTable::new();
    let n_blocks = ctx_len.div_ceil(block_size as usize);
    for _ in 0..n_blocks {
        pt.append_block(alloc)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    let n_kv_heads = preset.n_kv_heads;
    let n_q_heads = preset.n_q_heads;

    // Generate K, V for ctx_len tokens (simulating post-projection output)
    // Shape: [ctx_len, n_kv_heads, head_dim]
    let n_kv_elems = ctx_len * n_kv_heads as usize * head_dim as usize;
    let k_data = gen_tensor(n_kv_elems, seed, 0.5);
    let v_data = gen_tensor(n_kv_elems, seed + 1, 0.5);

    let k_buf = buffer_f16(device, &k_data);
    let v_buf = buffer_f16(device, &v_data);

    let bt_data: Vec<i32> = pt.as_slice().iter().map(|&x| x as i32).collect();
    let bt_buf = buffer_i32(device, &bt_data);

    // ── Step 1: write K, V into paged blocks ───────────────────────────
    let t_write = Instant::now();
    let cmd = new_cmd(queue);
    paged_attention::metal::dispatch_write_kv_to_blocks(
        pipelines,
        &cmd,
        &k_buf,
        0,
        &v_buf,
        0,
        alloc,
        layer,
        &bt_buf,
        ctx_len as u32,
        0, // start_pos
        n_kv_heads,
        head_dim,
        block_size,
    )?;
    cmd.commit();
    cmd.wait_until_completed();
    let write_ms = t_write.elapsed().as_secs_f64() * 1000.0;

    // ── Step 2: run paged decode ───────────────────────────────────────
    let q_data = gen_tensor(n_q_heads as usize * head_dim as usize, seed + 2, 1.0);
    let q_buf = buffer_f32(device, &q_data);

    let ctx_lens: Vec<i32> = vec![ctx_len as i32];
    let ctx_buf = buffer_i32(device, &ctx_lens);

    let max_num_blocks = n_blocks as u32;
    let bt_padded: Vec<i32> = pt.as_slice().iter().map(|&x| x as i32).collect();
    let bt_padded_buf = buffer_i32(device, &bt_padded);

    let output = device
        .new_buffer(
            n_q_heads as usize * head_dim as usize * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("output buffer");

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Which kernel: v1 (3-pass) or v2 (FlashAttention)
    let use_v2 = std::env::var("KERNEL").map(|s| s != "v1").unwrap_or(true);

    // Warmup to avoid first-dispatch cost
    for _ in 0..2 {
        let cmd = new_cmd(queue);
        if use_v2 {
            paged_attention::metal::dispatch_paged_attention_decode_v2(
                pipelines,
                &cmd,
                &q_buf,
                0,
                alloc,
                layer,
                &bt_padded_buf,
                &ctx_buf,
                &output,
                0,
                1,
                n_q_heads,
                n_kv_heads,
                head_dim,
                block_size,
                max_num_blocks,
                scale,
            )?;
        } else {
            paged_attention::metal::dispatch_paged_attention_decode(
                pipelines,
                &cmd,
                &q_buf,
                0,
                alloc,
                layer,
                &bt_padded_buf,
                &ctx_buf,
                &output,
                0,
                1,
                n_q_heads,
                n_kv_heads,
                head_dim,
                block_size,
                max_num_blocks,
                scale,
            )?;
        }
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Timed runs (median of 5)
    let n_iters = 5;
    let mut latencies = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t = Instant::now();
        let cmd = new_cmd(queue);
        if use_v2 {
            paged_attention::metal::dispatch_paged_attention_decode_v2(
                pipelines,
                &cmd,
                &q_buf,
                0,
                alloc,
                layer,
                &bt_padded_buf,
                &ctx_buf,
                &output,
                0,
                1,
                n_q_heads,
                n_kv_heads,
                head_dim,
                block_size,
                max_num_blocks,
                scale,
            )?;
        } else {
            paged_attention::metal::dispatch_paged_attention_decode(
                pipelines,
                &cmd,
                &q_buf,
                0,
                alloc,
                layer,
                &bt_padded_buf,
                &ctx_buf,
                &output,
                0,
                1,
                n_q_heads,
                n_kv_heads,
                head_dim,
                block_size,
                max_num_blocks,
                scale,
            )?;
        }
        cmd.commit();
        cmd.wait_until_completed();
        latencies.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let decode_ms = latencies[latencies.len() / 2];

    let gpu_out = read_f32(&output, n_q_heads as usize * head_dim as usize);

    // ── Step 3: CPU reference ──────────────────────────────────────────
    // For each q_head, compute reference attention using its mapped kv_head
    let gqa_ratio = n_q_heads / n_kv_heads;
    let mut max_diff: f32 = 0.0;
    let mut sum_rel: f32 = 0.0;
    let mut count_rel: usize = 0;

    for q_head in 0..n_q_heads {
        let kv_head = q_head / gqa_ratio;

        // Extract K and V for this kv_head: [ctx_len, head_dim]
        let mut k_head = Vec::with_capacity(ctx_len * head_dim as usize);
        let mut v_head = Vec::with_capacity(ctx_len * head_dim as usize);
        for t in 0..ctx_len {
            let base = (t * n_kv_heads as usize + kv_head as usize) * head_dim as usize;
            k_head.extend_from_slice(&k_data[base..base + head_dim as usize]);
            v_head.extend_from_slice(&v_data[base..base + head_dim as usize]);
        }

        // Quantize K, V to f16 and back (matches GPU precision)
        let k_head: Vec<f32> = k_head
            .iter()
            .map(|&x| f16_bits_to_f32(f32_to_f16_bits(x)))
            .collect();
        let v_head: Vec<f32> = v_head
            .iter()
            .map(|&x| f16_bits_to_f32(f32_to_f16_bits(x)))
            .collect();

        let q_head_data = &q_data
            [(q_head as usize * head_dim as usize)..((q_head + 1) as usize * head_dim as usize)];

        let ref_out = reference_attention(
            q_head_data,
            &k_head,
            &v_head,
            ctx_len,
            head_dim as usize,
            scale,
        );

        let gpu_head = &gpu_out
            [(q_head as usize * head_dim as usize)..((q_head + 1) as usize * head_dim as usize)];

        for (_i, (&g, &r)) in gpu_head.iter().zip(ref_out.iter()).enumerate() {
            let diff = (g - r).abs();
            max_diff = max_diff.max(diff);
            if r.abs() > 1e-4 {
                sum_rel += diff / r.abs();
                count_rel += 1;
            }
        }
    }

    let mean_rel = if count_rel > 0 {
        sum_rel / count_rel as f32
    } else {
        0.0
    };

    // Free blocks for reuse in next layer test
    pt.free_all(alloc);

    Ok((max_diff, mean_rel, write_ms, decode_ms))
}

fn main() -> Result<()> {
    let preset = match std::env::var("MODEL").as_deref() {
        Ok("31B") | Ok("31b") => &G31B,
        Ok("E4B") | Ok("e4b") => &E4B,
        _ => &G31B, // default: 31B
    };
    let ctx_len: usize = std::env::var("CTX_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let block_size: u32 = 16;

    eprintln!("╔════════════════════════════════════════════════════════════╗");
    eprintln!(
        "║ PagedAttention Real-Scale Test: Gemma 4 {} (ctx={})         ",
        preset.name, ctx_len
    );
    eprintln!("╚════════════════════════════════════════════════════════════╝");
    eprintln!(
        "  {} layers, n_q_heads={}, n_kv_heads={}, head_dim={}/{} (sliding/global)",
        preset.n_layers,
        preset.n_q_heads,
        preset.n_kv_heads,
        preset.head_dim_sliding,
        preset.head_dim_global,
    );
    let kernel_ver = std::env::var("KERNEL").unwrap_or_else(|_| "v2".into());
    eprintln!(
        "  Kernel: {} ({})",
        kernel_ver,
        if kernel_ver == "v1" {
            "3-pass naive"
        } else {
            "FlashAttention-style 1-pass"
        }
    );

    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().expect("command queue");
    let pipelines = PagedAttentionPipelines::new(&device)?;

    let config = build_config(preset, block_size);
    // Budget: enough for ctx_len tokens across ALL layers (+ some slack)
    let n_blocks_needed = ctx_len.div_ceil(block_size as usize) * 2;
    let budget = config.block_bytes_total() * n_blocks_needed;
    let mut alloc = BlockAllocator::new(&device, budget, config)?;

    eprintln!();
    eprintln!(
        "{:<6} {:<12} {:>9} {:>12} {:>11} {:>11}",
        "Layer", "Type", "HeadDim", "MaxAbsDiff", "MeanRelDiff", "Latency(ms)"
    );
    eprintln!("{:-<65}", "");

    // Sample a few layers across the model (first sliding, one global, last sliding)
    let n_layers = preset.n_layers as usize;
    let layer_indices = [0, n_layers / 2, n_layers - 1];

    let mut total_write_ms = 0.0;
    let mut total_decode_ms = 0.0;
    let mut worst_max_diff: f32 = 0.0;
    let mut worst_mean_rel: f32 = 0.0;

    for &layer in &layer_indices {
        let lc = &alloc.config.layer_configs[layer].clone();
        let (max_d, mean_rel, write_ms, decode_ms) = test_layer(
            &device,
            &queue,
            &pipelines,
            &mut alloc,
            layer,
            preset,
            ctx_len,
            block_size,
            lc.head_dim,
            42 + layer as u64,
        )?;

        eprintln!(
            "{:<6} {:<12} {:>9} {:>12.2e} {:>11.2e} {:>5.2}+{:<5.2}",
            layer,
            if lc.is_sliding { "sliding" } else { "global" },
            lc.head_dim,
            max_d,
            mean_rel,
            write_ms,
            decode_ms,
        );

        total_write_ms += write_ms;
        total_decode_ms += decode_ms;
        worst_max_diff = worst_max_diff.max(max_d);
        worst_mean_rel = worst_mean_rel.max(mean_rel);
    }

    eprintln!("{:-<65}", "");
    eprintln!();
    eprintln!("Summary:");
    eprintln!(
        "  Worst max abs diff:    {:.3e}  {}",
        worst_max_diff,
        if worst_max_diff < 1e-2 {
            "✓ PASS (<1e-2)"
        } else {
            "✗ FAIL"
        }
    );
    eprintln!(
        "  Worst mean rel diff:   {:.3e}  {}",
        worst_mean_rel,
        if worst_mean_rel < 5e-2 {
            "✓ PASS (<5%)"
        } else {
            "✗ FAIL"
        }
    );
    eprintln!(
        "  Avg write latency:     {:.2} ms / layer / {} tokens",
        total_write_ms / layer_indices.len() as f64,
        ctx_len
    );
    eprintln!(
        "  Avg decode latency:    {:.2} ms / layer (single-token)",
        total_decode_ms / layer_indices.len() as f64
    );

    // Extrapolate full-model decode latency
    let avg_decode_single = total_decode_ms / layer_indices.len() as f64;
    let full_model_decode = avg_decode_single * preset.n_layers as f64;
    let tok_per_s = 1000.0 / full_model_decode;
    eprintln!(
        "  Extrapolated full model: {:.2} ms/tok ≈ {:.1} tok/s (decode only)",
        full_model_decode, tok_per_s
    );

    Ok(())
}
