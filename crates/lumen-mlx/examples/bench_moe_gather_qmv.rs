//! Microbench `mlx::gather_qmm` (= `affine_gather_qmv_fast` at decode) at
//! Gemma 4 26B-A4B MoE shapes. Step 0b of decode BW headroom plan.
//!
//! Decode MoE per layer:
//!   x:           [B=1, L=1, K=8, hidden=2816]   bf16
//!   gate/up w:   [E=128, intermediate=704, hidden=2816]    Q4 affine, group_size=64
//!   down w:      [E=128, hidden=2816, intermediate=704]    Q4 affine, group_size=64
//!   indices:     [B=1, L=1, K=8]   uint32 in [0, 128)
//!
//! Reads per call (8 experts selected at decode):
//!   gate: 8 * 704 * 352 * 4 (packed u32) + 2 * 8 * 704 * 44 * 2 (scales+biases bf16)
//!        = 7.9 MB + 0.5 MB + 0.5 MB ≈ 8.9 MB
//!   up:   same ≈ 8.9 MB
//!   down: same shape transposed ≈ 8.9 MB
//!
//! Each call's theoretical BW @ 220 GB/s effective:
//!   t = 8.9 MB / 220 GB/s = 40.5 μs
//!
//! Three runs (gate, up, down) per MoE layer × 25 MoE layers per token
//!   = 75 dispatches per decode token
//!   = 0.66 GB read per token just for MoE
//!   = 3 ms @ 220 GB/s for MoE alone (at 80 tok/s pace → 12.5 ms budget per token,
//!     so MoE 3ms = 24% — matches earlier 29% breakdown)
//!
//! Usage:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx cargo run --release \
//!     --features mlx-native-metal --example bench_moe_gather_qmv \
//!     -- [iters]
//!
//! Optional env:
//!   MOE_BENCH_HIDDEN=2816       Default Gemma 4 26B-A4B hidden_size
//!   MOE_BENCH_INTERMEDIATE=704  Default Gemma 4 26B-A4B moe_intermediate_size
//!   MOE_BENCH_EXPERTS=128       Default num_experts
//!   MOE_BENCH_TOP_K=8           Default top_k_experts
//!   MOE_BENCH_ITERS=200         Override timed iterations
//!   MOE_BENCH_WARMUP=20         Override warmup iterations
//!   MOE_BENCH_CHAINS_PER_EVAL=1 Chains queued before each eval (1 = sync per
//!                               chain, 75 = real per-token MoE pattern: 25
//!                               layers × 3 dispatches before final sync)
//!   MOE_BENCH_ROTATE_INDICES=0  Rotate expert indices each chain (1 = mimics
//!                               production cold-cache pattern, 0 = same 8
//!                               experts every call = warm L2)

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use mlx_rs::ops::{gather_qmm, quantize};
    use mlx_rs::random;
    use mlx_rs::{Array, Dtype};
    use std::time::Instant;

    let hidden: i32 = std::env::var("MOE_BENCH_HIDDEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2816);
    let intermediate: i32 = std::env::var("MOE_BENCH_INTERMEDIATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(704);
    let experts: i32 = std::env::var("MOE_BENCH_EXPERTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let top_k: i32 = std::env::var("MOE_BENCH_TOP_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let iters: usize = std::env::var("MOE_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let warmup: usize = std::env::var("MOE_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let chains_per_eval: usize = std::env::var("MOE_BENCH_CHAINS_PER_EVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let rotate_indices: bool = std::env::var("MOE_BENCH_ROTATE_INDICES")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);

    let group_size: i32 = 64;
    let bits: i32 = 4;

    eprintln!(
        "[bench-moe-gather-qmv] hidden={hidden} intermediate={intermediate} experts={experts} top_k={top_k} iters={iters} warmup={warmup} chains_per_eval={chains_per_eval} rotate_indices={rotate_indices}"
    );

    // Generate quantized weights for gate/up/down. Shape convention from
    // mlx_lm: weight stored as [E, out, in] with quantize along last axis.
    // gate/up: out=intermediate, in=hidden
    // down:    out=hidden,       in=intermediate
    let gate_full =
        random::normal::<f32>(&[experts, intermediate, hidden], None, None, None)?
            .as_dtype(Dtype::Bfloat16)
            .context("gate as_dtype")?;
    let up_full = random::normal::<f32>(&[experts, intermediate, hidden], None, None, None)?
        .as_dtype(Dtype::Bfloat16)
        .context("up as_dtype")?;
    let down_full =
        random::normal::<f32>(&[experts, hidden, intermediate], None, None, None)?
            .as_dtype(Dtype::Bfloat16)
            .context("down as_dtype")?;

    let (gate_w, gate_s, gate_b) = quantize(&gate_full, group_size, bits)?;
    let (up_w, up_s, up_b) = quantize(&up_full, group_size, bits)?;
    let (down_w, down_s, down_b) = quantize(&down_full, group_size, bits)?;

    // Eagerly evaluate so the timed loop doesn't include allocation/compile.
    gate_w.eval()?; gate_s.eval()?; gate_b.eval()?;
    up_w.eval()?;   up_s.eval()?;   up_b.eval()?;
    down_w.eval()?; down_s.eval()?; down_b.eval()?;

    let bytes_per_call: u64 = {
        // packed last-axis dim = in / (32 / bits) for bits=4 → in/8 uint32
        let in_dim = hidden as u64;
        let out_dim = intermediate as u64;
        // gate/up bytes:
        let packed_bytes = (top_k as u64) * out_dim * (in_dim / 8) * 4;
        let scales_bytes = (top_k as u64) * out_dim * (in_dim / group_size as u64) * 2;
        let biases_bytes = (top_k as u64) * out_dim * (in_dim / group_size as u64) * 2;
        packed_bytes + scales_bytes + biases_bytes
    };
    let mb_per_call = bytes_per_call as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[bench-moe-gather-qmv] approx bytes/call (gate or up): {mb_per_call:.2} MB"
    );

    // Decode-shape input: [B=1, L=1, K=top_k, hidden]
    let x_in = random::normal::<f32>(&[1, 1, top_k, hidden], None, None, None)?
        .as_dtype(Dtype::Bfloat16)
        .context("x as_dtype")?;
    x_in.eval()?;

    // Indices pool: when rotating, generate N distinct top_k selections to
    // mimic production decode (different 8 experts per layer per token).
    // When static, all entries are the same baseline selection.
    let pool_size: usize = if rotate_indices { 128 } else { 1 };
    let mut indices_pool: Vec<Array> = Vec::with_capacity(pool_size);
    for p in 0..pool_size {
        let seq: Vec<u32> = if rotate_indices {
            // Rotate the K=top_k window through [0, experts) so each chain
            // touches a disjoint slab of expert weights when feasible.
            let start = ((p * top_k as usize) % experts as usize) as u32;
            (0..top_k as u32).map(|i| (start + i) % experts as u32).collect()
        } else {
            (0..top_k as u32).collect()
        };
        let arr = Array::from_slice(&seq, &[1, 1, top_k])
            .as_dtype(Dtype::Uint32)
            .context("indices as_dtype")?;
        arr.eval()?;
        indices_pool.push(arr);
    }

    // Run one chain (gate → up → activation simulated by * → down) per iter.
    // Match the dispatch the real MoE forward issues per layer.
    let run_chain = |indices: &Array| -> anyhow::Result<Array> {
        let x_gate = gather_qmm(
            &x_in,
            &gate_w,
            &gate_s,
            Some(&gate_b),
            None,
            Some(indices),
            /* transpose */ true,
            /* group_size */ group_size,
            /* bits */ bits,
            /* sorted_indices */ false,
        )
        .context("gather_qmm gate")?;
        let x_up = gather_qmm(
            &x_in,
            &up_w,
            &up_s,
            Some(&up_b),
            None,
            Some(indices),
            true,
            group_size,
            bits,
            false,
        )
        .context("gather_qmm up")?;
        let activated = x_gate.multiply(&x_up).context("multiply gate*up")?;
        let down = gather_qmm(
            &activated,
            &down_w,
            &down_s,
            Some(&down_b),
            None,
            Some(indices),
            true,
            group_size,
            bits,
            false,
        )
        .context("gather_qmm down")?;
        Ok(down)
    };

    // Accumulator helper: sum all chain outputs into one Array, then eval the
    // sum. This forces mlx to compute every queued chain (otherwise eval on the
    // last chain alone would let the prior chains' subgraphs be dropped).
    let mut chain_counter: usize = 0;
    let mut run_group = |n: usize| -> anyhow::Result<()> {
        let mut acc: Option<Array> = None;
        for _ in 0..n {
            let idx_ref = &indices_pool[chain_counter % indices_pool.len()];
            chain_counter = chain_counter.wrapping_add(1);
            let y = run_chain(idx_ref)?;
            acc = Some(match acc {
                Some(a) => a.add(&y).context("acc add")?,
                None => y,
            });
        }
        if let Some(a) = acc {
            a.eval()?;
        }
        Ok(())
    };

    // Warmup.
    let warmup_groups = warmup.div_ceil(chains_per_eval.max(1));
    for _ in 0..warmup_groups {
        run_group(chains_per_eval)?;
    }

    // Timed loop. Queue `chains_per_eval` chains before forcing a sync — this
    // amortizes launch/dispatch overhead the same way a real decode step does
    // (75 dispatches/token for 25 MoE layers × 3 ops, all evaluated together).
    let groups = iters.div_ceil(chains_per_eval.max(1));
    let total_chains = groups * chains_per_eval;
    let t0 = Instant::now();
    for _ in 0..groups {
        run_group(chains_per_eval)?;
    }
    let elapsed = t0.elapsed();

    let ns_per_iter = elapsed.as_nanos() as f64 / total_chains as f64;
    let bytes_per_iter = bytes_per_call * 3;
    let gbps = bytes_per_iter as f64 / (ns_per_iter); // bytes/ns == GB/s
    let us_per_iter = ns_per_iter / 1000.0;

    println!("=== Microbench: MoE gather_qmm chain (gate + up + down) ===");
    println!("  iters:            {iters} (chains: {total_chains}, groups: {groups})");
    println!("  warmup:           {warmup}");
    println!("  chains_per_eval:  {chains_per_eval}");
    println!("  shape:            B=1 L=1 K={top_k} hidden={hidden} intermediate={intermediate} experts={experts}");
    println!("  bytes/chain:      {:.2} MB ({} bytes)", (bytes_per_iter as f64) / (1024.0 * 1024.0), bytes_per_iter);
    println!("  time/chain:       {us_per_iter:.2} μs");
    println!("  effective BW:     {gbps:.1} GB/s");
    println!();
    println!("Reference for 8K decode budget:");
    println!("  MoE 25 layers × {us_per_iter:.2} μs = {:.2} ms per token", us_per_iter * 25.0 / 1000.0);
    println!("  vs current 8K decode 17.5 ms → MoE = {:.1}%",
        (us_per_iter * 25.0 / 1000.0 / 17.5) * 100.0);

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("bench_moe_gather_qmv requires --features mlx-native(-metal)")
}
