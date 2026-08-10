//! Integration test for PagedKVBackend.
//!
//! Creates a backend in-process, feeds it Candle Metal tensors with known
//! values, and verifies that store_kv + compressed_attention produce a
//! numerically correct result (matches a CPU reference SDPA).
//!
//! Two contract details this test got wrong for a long time, both of which
//! made it report a max_diff of ~0.7 against a kernel that is exact:
//!
//! - **`store_kv` takes the whole cache, not the new tail.** The model calls
//!   it with what `kv_cache.append` returned, and the backend narrows
//!   `seq_len_before..cache_seq_len` itself. Handing it only the new token
//!   makes `cache_seq_len < seq_len`, which the backend reads as a cache
//!   reset: it rotates to a fresh sequence, returns `true`, and silently
//!   drops the history. Attention then runs over 1 token instead of 5.
//! - **The attention scale at this call site is 1.0.** Gemma 4 folds the
//!   query scale into the weights; the SDPA fallback sitting right next to
//!   `compressed_attention` in `quantized_gemma4` passes 1.0 too.
//!
//! With both fixed the kernel matches the CPU reference to 9e-5.

#![cfg(feature = "paged-kv")]

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::CompressedKVBackend;
use lumen_model::paged_kv::PagedKVBackend;

/// CPU reference SDPA for one head (f32 math).
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ctx_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(ctx_len);
    for t in 0..ctx_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[t * head_dim + d];
        }
        scores.push(dot * scale);
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let mut out = vec![0.0f32; head_dim];
    for t in 0..ctx_len {
        for d in 0..head_dim {
            out[d] += probs[t] * v[t * head_dim + d];
        }
    }
    out
}

/// Deterministic pseudo-random f32 in ~[-1, 1].
fn rand_f32(seed: u64) -> f32 {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    ((s as i32) as f32) / (i32::MAX as f32)
}

fn gen_tensor(shape: &[usize], seed: u64, scale: f32, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| rand_f32(seed.wrapping_add(i as u64)) * scale)
        .collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}

/// Prefill store + single decode attention.
/// Uses uniform tiny dims for quick verification.
#[test]
fn test_paged_kv_backend_end_to_end() -> Result<()> {
    // Tiny model — 2 layers (both sliding), n_kv=2, head_dim=64
    let n_layers: u32 = 2;
    let n_kv_heads: u32 = 2;
    let n_q_heads: u32 = 2; // no GQA for this test
    let head_dim: u32 = 64;
    let block_size: u32 = 16;

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping: no Metal device");
            return Ok(());
        }
    };

    let mut backend = PagedKVBackend::new(
        device.clone(),
        32, // 32 MB budget
        block_size,
        n_layers,
        n_kv_heads,
        head_dim, // head_dim_sliding
        head_dim, // head_dim_global (same for simplicity)
        999,      // global_every=999 → all sliding
    )?;

    // Prefill: 4 tokens
    let prefill_len: usize = 4;
    let k_prefill = gen_tensor(
        &[1, n_kv_heads as usize, prefill_len, head_dim as usize],
        100,
        0.5,
        &device,
    )?;
    let v_prefill = gen_tensor(
        &[1, n_kv_heads as usize, prefill_len, head_dim as usize],
        200,
        0.5,
        &device,
    )?;

    // Write prefill for layer 0
    let stored = backend.store_kv(0, &k_prefill, &v_prefill);
    assert!(stored, "store_kv for layer 0 prefill failed");
    // Write prefill for layer 1
    let stored = backend.store_kv(1, &k_prefill, &v_prefill);
    assert!(stored, "store_kv for layer 1 prefill failed");

    // Decode: 1 new token, for layer 0 first
    let k_dec = gen_tensor(
        &[1, n_kv_heads as usize, 1, head_dim as usize],
        300,
        0.5,
        &device,
    )?;
    let v_dec = gen_tensor(
        &[1, n_kv_heads as usize, 1, head_dim as usize],
        400,
        0.5,
        &device,
    )?;

    // The decode store gets the FULL cache — prefill history plus the new
    // token — because that is what `kv_cache.append` hands the model.
    let k_all = Tensor::cat(&[&k_prefill, &k_dec], 2)?.contiguous()?;
    let v_all = Tensor::cat(&[&v_prefill, &v_dec], 2)?.contiguous()?;
    let stored = backend.store_kv(0, &k_all, &v_all);
    assert!(stored, "store_kv for layer 0 decode failed");
    // `store_kv` returns true both when it appends and when it decides the
    // cache was reset and starts over, so the return value alone does not
    // say the history survived. This does.
    assert_eq!(
        backend.seq_len(0),
        prefill_len + 1,
        "layer 0 lost its prefill history on the decode store"
    );

    // Query for layer 0
    let q = gen_tensor(
        &[1, n_q_heads as usize, 1, head_dim as usize],
        500,
        0.5,
        &device,
    )?;

    let gpu_out = backend
        .compressed_attention(
            0,
            &q,
            n_q_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
        )
        .expect("compressed_attention returned None");

    // Convert output to CPU f32 Vec
    let gpu_out_vec: Vec<f32> = gpu_out.flatten_all()?.to_vec1::<f32>()?;

    // Sanity: no NaN
    let nan_count = gpu_out_vec.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "GPU output contains {} NaN values", nan_count);

    // Reference computation: build [5, head_dim] K, V per kv_head from prefill + decode
    let ctx_len = prefill_len + 1; // 5
    let k_prefill_vec: Vec<f32> = k_prefill.flatten_all()?.to_vec1::<f32>()?;
    let v_prefill_vec: Vec<f32> = v_prefill.flatten_all()?.to_vec1::<f32>()?;
    let k_dec_vec: Vec<f32> = k_dec.flatten_all()?.to_vec1::<f32>()?;
    let v_dec_vec: Vec<f32> = v_dec.flatten_all()?.to_vec1::<f32>()?;
    let q_vec: Vec<f32> = q.flatten_all()?.to_vec1::<f32>()?;

    // Candle layout: [1, n_kv, seq, dim] — flat index (kv_h * seq + t) * dim + d
    let hd = head_dim as usize;
    let nkv = n_kv_heads as usize;
    let nq = n_q_heads as usize;
    let gqa = nq / nkv;
    // See the module comment: the query scale is already folded into the
    // weights by the time attention runs, so this call site uses 1.0.
    let scale = 1.0f32;

    let mut max_diff: f32 = 0.0;
    for q_head in 0..nq {
        let kv_head = q_head / gqa;

        // Build [ctx_len, head_dim] K and V for this kv_head
        let mut k_head = Vec::with_capacity(ctx_len * hd);
        let mut v_head = Vec::with_capacity(ctx_len * hd);
        for t in 0..prefill_len {
            let base = (kv_head * prefill_len + t) * hd;
            k_head.extend_from_slice(&k_prefill_vec[base..base + hd]);
            v_head.extend_from_slice(&v_prefill_vec[base..base + hd]);
        }
        let dec_base = kv_head * hd;
        k_head.extend_from_slice(&k_dec_vec[dec_base..dec_base + hd]);
        v_head.extend_from_slice(&v_dec_vec[dec_base..dec_base + hd]);

        // No f16 round-trip here: `store_kv` casts to f32 and the paged
        // blocks hold f32 (`dispatch_write_kv_f32_candle`). Rounding the
        // reference to f16 would be modelling a different backend.
        let q_slice = &q_vec[(q_head * hd)..((q_head + 1) * hd)];

        let ref_out = reference_attention(q_slice, &k_head, &v_head, ctx_len, hd, scale);

        let gpu_slice = &gpu_out_vec[(q_head * hd)..((q_head + 1) * hd)];
        let mut head_max: f32 = 0.0;
        for (g, r) in gpu_slice.iter().zip(ref_out.iter()) {
            head_max = head_max.max((g - r).abs());
        }
        eprintln!(
            "  head {}: kv_head={} max_diff={:.6e} ref[0..4]={:.3?} gpu[0..4]={:.3?}",
            q_head,
            kv_head,
            head_max,
            &ref_out[0..4.min(ref_out.len())],
            &gpu_slice[0..4.min(gpu_slice.len())]
        );
        max_diff = max_diff.max(head_max);
    }

    eprintln!("paged_kv end-to-end max_diff = {:.6e}", max_diff);
    // 1e-2 was loose enough to accept attention over the wrong context (a
    // one-token context lands at 8e-1, but a dropped *first* token lands at
    // 1.8e-1 and a scale error at 2.3e-1 — none far from a threshold chosen
    // for f16). The f32 path reproduces the reference to 9e-5.
    assert!(max_diff < 1e-3, "max diff {max_diff:.6e} too large (>1e-3)");

    Ok(())
}
