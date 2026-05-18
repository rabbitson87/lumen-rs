//! Integration tests for PagedAttention Metal kernels.
//!
//! Tests:
//! 1. write_kv_to_blocks: write KV data, read back, verify correct placement
//! 2. paged_attention_decode: single query attention against paged KV, compare with reference
//! 3. copy_blocks: block copy correctness

use std::sync::Arc;

use candle_metal_kernels::metal::{
    Buffer, CommandBuffer, CommandQueue, CommandSemaphore, Device, MTLResourceOptions,
    create_command_buffer,
};
use paged_attention::block_allocator::{BlockAllocator, BlockConfig, LayerAttentionConfig};
use paged_attention::block_table::PageTable;
use paged_attention::metal::dispatch::PagedAttentionPipelines;

/// Helper: create Metal device or skip test.
fn metal_device() -> Option<Device> {
    Device::system_default()
}

fn new_cmd(queue: &CommandQueue) -> CommandBuffer {
    create_command_buffer(queue, Arc::new(CommandSemaphore::new())).expect("cmd buffer")
}

/// Helper: create a buffer with f16 data from f32 source.
fn buffer_f16(device: &Device, data: &[f32]) -> Buffer {
    let f16_data: Vec<u16> = data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    device
        .new_buffer_with_data(
            f16_data.as_ptr() as *const _,
            f16_data.len() * 2,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
}

/// Helper: create a buffer with f32 data.
fn buffer_f32(device: &Device, data: &[f32]) -> Buffer {
    device
        .new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
}

/// Helper: create a buffer with i32 data.
fn buffer_i32(device: &Device, data: &[i32]) -> Buffer {
    device
        .new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
}

/// Helper: read f16 buffer back as f32.
fn read_f16(buf: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const u16;
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    slice
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect()
}

/// Helper: read f32 buffer.
fn read_f32(buf: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    slice.to_vec()
}

// ============================================================================
// Test 1: write_kv_to_blocks
// ============================================================================

#[test]
fn test_write_kv_to_blocks() {
    let device = match metal_device() {
        Some(d) => d,
        None => return,
    };
    let queue = device.new_command_queue().unwrap();
    let pipelines = PagedAttentionPipelines::new(&device).unwrap();

    // Config: 1 layer, 2 kv_heads, head_dim=4, block_size=4
    let config = BlockConfig::uniform(4, 1, 2, 4);
    let budget = config.block_bytes_total() * 10;
    let mut alloc = BlockAllocator::new(&device, budget, config).unwrap();

    // Allocate 2 blocks for a sequence of 5 tokens
    let mut pt = PageTable::new();
    pt.append_block(&mut alloc).unwrap(); // block for tokens 0-3
    pt.append_block(&mut alloc).unwrap(); // block for token 4

    // K data: 5 tokens × 2 heads × 4 dim = 40 elements
    // Pattern: k[t][h][d] = t*100 + h*10 + d
    let mut k_data = vec![0.0f32; 5 * 2 * 4];
    let mut v_data = vec![0.0f32; 5 * 2 * 4];
    for t in 0..5u32 {
        for h in 0..2u32 {
            for d in 0..4u32 {
                let idx = (t * 2 * 4 + h * 4 + d) as usize;
                k_data[idx] = (t * 100 + h * 10 + d) as f32;
                v_data[idx] = (t * 100 + h * 10 + d + 1000) as f32;
            }
        }
    }

    let k_buf = buffer_f16(&device, &k_data);
    let v_buf = buffer_f16(&device, &v_data);
    let bt_data: Vec<i32> = pt.as_slice().iter().map(|&x| x as i32).collect();
    let bt_buf = buffer_i32(&device, &bt_data);

    let cmd = new_cmd(&queue);
    paged_attention::metal::dispatch_write_kv_to_blocks(
        &pipelines, &cmd, &k_buf, 0, &v_buf, 0, &alloc, 0, // layer 0
        &bt_buf, 5, // seq_len
        0, // start_pos
        2, // n_kv_heads
        4, // head_dim
        4, // block_size
    )
    .unwrap();
    cmd.commit();
    cmd.wait_until_completed();

    // Read back from KV pool and verify
    let layer_buf = alloc.layer_buffer(0);
    let block_bytes = alloc.layer_block_bytes(0);
    let block_elems = block_bytes / 2; // f16 = 2 bytes each

    // Read block 0: should contain tokens 0-3
    let _block0 = read_f16(layer_buf, block_elems);
    // K section: first block_size * n_kv_heads * head_dim = 4*2*4 = 32 elements
    let kv_section_size = 4 * 2 * 4; // block_size * n_kv_heads * head_dim

    // Verify token 0, head 0, dim 0 in K section
    let phys0 = pt.get(0) as usize;
    let k_start = phys0 * block_elems;
    // Since we read the entire buffer, index directly
    let pool_data = read_f16(layer_buf, alloc.num_total_blocks() as usize * block_elems);

    // Token 0, head 0: K should be [0, 1, 2, 3]
    for d in 0..4 {
        let idx = k_start + 0 * 2 * 4 + 0 * 4 + d; // token=0, head=0
        let expected = d as f32;
        let actual = pool_data[idx];
        assert!(
            (actual - expected).abs() < 0.1,
            "K[t=0,h=0,d={d}]: expected {expected}, got {actual}"
        );
    }

    // Token 2, head 1: K should be [210, 211, 212, 213]
    for d in 0..4 {
        let idx = k_start + 2 * 2 * 4 + 1 * 4 + d;
        let expected = (200 + 10 + d) as f32;
        let actual = pool_data[idx];
        assert!(
            (actual - expected).abs() < 0.1,
            "K[t=2,h=1,d={d}]: expected {expected}, got {actual}"
        );
    }

    // Token 0, head 0: V should be [1000, 1001, 1002, 1003]
    let v_start = k_start + kv_section_size;
    for d in 0..4 {
        let idx = v_start + 0 * 2 * 4 + 0 * 4 + d;
        let expected = (1000 + d) as f32;
        let actual = pool_data[idx];
        assert!(
            (actual - expected).abs() < 0.1,
            "V[t=0,h=0,d={d}]: expected {expected}, got {actual}"
        );
    }

    // Token 4 is in block 1
    let phys1 = pt.get(1) as usize;
    let k_start_b1 = phys1 * block_elems;
    // Token 4 (offset 0 within block 1), head 0: K should be [400, 401, 402, 403]
    for d in 0..4 {
        let idx = k_start_b1 + 0 * 2 * 4 + 0 * 4 + d;
        let expected = (400 + d) as f32;
        let actual = pool_data[idx];
        assert!(
            (actual - expected).abs() < 0.1,
            "K[t=4,h=0,d={d}] in block1: expected {expected}, got {actual}"
        );
    }
}

// ============================================================================
// Test 2: paged_attention_decode
// ============================================================================

#[test]
fn test_paged_attention_decode_simple() {
    let device = match metal_device() {
        Some(d) => d,
        None => return,
    };
    let queue = device.new_command_queue().unwrap();
    let pipelines = PagedAttentionPipelines::new(&device).unwrap();

    // Config: 1 layer, 1 kv_head, head_dim=4, block_size=4
    let n_kv_heads: u32 = 1;
    let head_dim: u32 = 4;
    let block_size: u32 = 4;
    let config = BlockConfig::uniform(block_size, 1, n_kv_heads, head_dim);
    let budget = config.block_bytes_total() * 10;
    let mut alloc = BlockAllocator::new(&device, budget, config).unwrap();

    // Setup: 3 tokens in KV cache
    let mut pt = PageTable::new();
    pt.append_block(&mut alloc).unwrap();

    // K vectors: [1,0,0,0], [0,1,0,0], [0,0,1,0]
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    // V vectors: [1,0,0,0], [0,1,0,0], [0,0,1,0]
    let v_data: Vec<f32> = k_data.clone();

    let k_buf = buffer_f16(&device, &k_data);
    let v_buf = buffer_f16(&device, &v_data);
    let bt_data: Vec<i32> = pt.as_slice().iter().map(|&x| x as i32).collect();
    let bt_buf = buffer_i32(&device, &bt_data);

    // Write KV to blocks
    let cmd = new_cmd(&queue);
    paged_attention::metal::dispatch_write_kv_to_blocks(
        &pipelines, &cmd, &k_buf, 0, &v_buf, 0, &alloc, 0, &bt_buf, 3, 0, n_kv_heads, head_dim,
        block_size,
    )
    .unwrap();
    cmd.commit();
    cmd.wait_until_completed();

    // Query: [1, 0, 0, 0] — should attend mostly to first K token
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let q_buf = buffer_f32(&device, &q_data);

    let context_lens: Vec<i32> = vec![3];
    let ctx_buf = buffer_i32(&device, &context_lens);

    // Pad block table to max_num_blocks
    let max_num_blocks: u32 = 4;
    let mut bt_padded: Vec<i32> = vec![0; max_num_blocks as usize];
    for (i, &b) in pt.as_slice().iter().enumerate() {
        bt_padded[i] = b as i32;
    }
    let bt_padded_buf = buffer_i32(&device, &bt_padded);

    let output = device
        .new_buffer(
            head_dim as usize * 4, // f32
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();

    let n_q_heads: u32 = 1;
    let scale: f32 = 1.0; // no scaling for simple test

    let cmd = new_cmd(&queue);
    paged_attention::metal::dispatch_paged_attention_decode(
        &pipelines,
        &cmd,
        &q_buf,
        0,
        &alloc,
        0,
        &bt_padded_buf,
        &ctx_buf,
        &output,
        0,
        1, // batch_size
        n_q_heads,
        n_kv_heads,
        head_dim,
        block_size,
        max_num_blocks,
        scale,
    )
    .unwrap();
    cmd.commit();
    cmd.wait_until_completed();

    let result = read_f32(&output, head_dim as usize);

    // With q=[1,0,0,0], K=identity-like, scale=1.0:
    // scores = [1.0, 0.0, 0.0] → softmax ≈ [0.576, 0.212, 0.212]
    // output ≈ 0.576*[1,0,0,0] + 0.212*[0,1,0,0] + 0.212*[0,0,1,0]
    //        ≈ [0.576, 0.212, 0.212, 0.0]

    // Verify first dim is largest (highest attention to first token)
    assert!(
        result[0] > result[1],
        "result[0]={} should > result[1]={}",
        result[0],
        result[1]
    );
    assert!(
        result[0] > 0.4,
        "result[0]={} should be > 0.4 (dominant attention)",
        result[0]
    );
    // Sum should ≈ 1.0 (it's a weighted sum of unit-ish vectors)
    let sum: f32 = result.iter().sum();
    assert!(
        sum > 0.9 && sum < 1.1,
        "sum of output dims = {sum}, expected ≈ 1.0"
    );

    eprintln!("paged_attention_decode result: {:?}", result);
}

// ============================================================================
// Test 3: Heterogeneous layers (31B-like)
// ============================================================================

#[test]
fn test_write_kv_heterogeneous() {
    let device = match metal_device() {
        Some(d) => d,
        None => return,
    };
    let queue = device.new_command_queue().unwrap();
    let pipelines = PagedAttentionPipelines::new(&device).unwrap();

    // 2 layers: sliding(head_dim=4), global(head_dim=8)
    let layers = vec![
        LayerAttentionConfig {
            n_kv_heads: 1,
            head_dim: 4,
            is_sliding: true,
        },
        LayerAttentionConfig {
            n_kv_heads: 1,
            head_dim: 8,
            is_sliding: false,
        },
    ];
    let config = BlockConfig::from_layers(4, layers);
    let budget = config.block_bytes_total() * 10;
    let mut alloc = BlockAllocator::new(&device, budget, config.clone()).unwrap();

    let mut pt = PageTable::new();
    pt.append_block(&mut alloc).unwrap();

    // Layer 0: 2 tokens, 1 head, dim=4
    let k0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v0: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let k0_buf = buffer_f16(&device, &k0);
    let v0_buf = buffer_f16(&device, &v0);

    // Layer 1: 2 tokens, 1 head, dim=8
    let k1: Vec<f32> = (0..16).map(|i| (i + 100) as f32).collect();
    let v1: Vec<f32> = (0..16).map(|i| (i + 200) as f32).collect();
    let k1_buf = buffer_f16(&device, &k1);
    let v1_buf = buffer_f16(&device, &v1);

    let bt_data: Vec<i32> = pt.as_slice().iter().map(|&x| x as i32).collect();
    let bt_buf = buffer_i32(&device, &bt_data);

    let cmd = new_cmd(&queue);

    // Write layer 0 (dim=4)
    paged_attention::metal::dispatch_write_kv_to_blocks(
        &pipelines, &cmd, &k0_buf, 0, &v0_buf, 0, &alloc, 0, &bt_buf, 2, 0, 1, 4, 4,
    )
    .unwrap();

    // Write layer 1 (dim=8)
    paged_attention::metal::dispatch_write_kv_to_blocks(
        &pipelines, &cmd, &k1_buf, 0, &v1_buf, 0, &alloc, 1, &bt_buf, 2, 0, 1, 8, 4,
    )
    .unwrap();

    cmd.commit();
    cmd.wait_until_completed();

    // Verify layer 0 buffer (dim=4)
    let lb0 = alloc.layer_block_bytes(0);
    let phys = pt.get(0) as usize;
    let pool0 = read_f16(
        alloc.layer_buffer(0),
        alloc.num_total_blocks() as usize * lb0 / 2,
    );
    let block_start = phys * lb0 / 2;

    // K: token 0 = [1,2,3,4]
    assert!((pool0[block_start] - 1.0).abs() < 0.1, "layer0 K[0][0]");
    assert!((pool0[block_start + 3] - 4.0).abs() < 0.1, "layer0 K[0][3]");

    // Verify layer 1 buffer (dim=8)
    let lb1 = alloc.layer_block_bytes(1);
    let pool1 = read_f16(
        alloc.layer_buffer(1),
        alloc.num_total_blocks() as usize * lb1 / 2,
    );
    let block_start1 = phys * lb1 / 2;

    // K: token 0 = [100,101,...,107]
    assert!((pool1[block_start1] - 100.0).abs() < 0.1, "layer1 K[0][0]");
    assert!(
        (pool1[block_start1 + 7] - 107.0).abs() < 0.1,
        "layer1 K[0][7]"
    );

    eprintln!("Heterogeneous write_kv test passed: layer0(dim=4), layer1(dim=8)");
}
