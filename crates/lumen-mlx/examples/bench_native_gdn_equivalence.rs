//! Native-Rust standalone equivalence test for the `gated_delta_step` Metal
//! kernel — D5d prerequisite for D5a (DFlash native port).
//!
//! Question: does
//!     gated_delta_step(q[:N], k[:N], v[:N], g[:N], β[:N], state_in)
//!   ≡ N × gated_delta_step(q[i:i+1], …, state_chained)
//!
//! at the mlx-rs binding level (i.e., independent of the rest of the model
//! forward, conv1d, RMSNorm, q/k norm, out_proj, etc.)?
//!
//! If YES → DFlash native port preserves correctness; D5a is greenlit.
//! If NO  → native port would just rediscover the GDN replay drift; the
//!         dead-end shifts upstream to the kernel itself, and the path
//!         forward is to fork mlx-rs / patch the kernel dispatcher to
//!         restore equivalence (per user direction 2026-05-05).
//!
//! Method (kernel-only — no model, no DFlash control flow):
//!   1. Build random q[B,1,Hk,Dk], k[B,1,Hk,Dk], v[B,1,Hv,Dv],
//!      g[B,1,Hv]∈(0,1), β[B,1,Hv]∈(0,1) for each of n_max positions.
//!   2. For each N in n_list:
//!      a. Path A: concat(first N of each input along seq) →
//!      one kernel call → (y_A:[B,N,Hv,Dv], state_A:[B,Hv,Dv,Dk]).
//!      b. Path B: state := zeros; for i in 0..N: feed (q_i, k_i, v_i, g_i,
//!      β_i, state) → kernel → (y_i, state). Concat all y_i → y_B.
//!   3. Compare max-|y_A − y_B| and max-|state_A − state_B|.
//!
//! Bit-identical (max diff == 0) is the BAR; fp accumulation noise (1e-5..1e-3)
//! is acceptable but flagged. Sharp non-zero (≥ 1e-2) at any N proves the
//! kernel itself is not sequential-equivalent.
//!
//! Usage:
//!   cargo run --release -p lumen-mlx \
//!     --features mlx-native-metal \
//!     --example bench_native_gdn_equivalence -- \
//!     [--n-list 2,3,4,5,6,7,8,12,16,24,32]
//!
//! Default shape mirrors Qwen3.6 GDN dimensions where possible (Dk multiple of
//! 32; Dv multiple of 4 for thread-group y).

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use lumen_mlx::native_ssm::gated_delta_step_kernel;
    use mlx_rs::{Array, ops};

    // ── CLI ──────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_list_raw = String::from("2,3,4,5,6,7,8,12,16,24,32");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n-list" => {
                n_list_raw = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    let mut n_list: Vec<usize> = n_list_raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .collect();
    n_list.sort_unstable();
    n_list.dedup();
    if n_list.is_empty() {
        anyhow::bail!("no valid N values in --n-list={n_list_raw:?}");
    }
    let n_max = *n_list.last().expect("non-empty");

    // ── Shape ────────────────────────────────────────────────────────────────
    // Kernel constraints: Dk multiple of 32 (simdgroup tile), B*Hv*Dv reasonable
    // for thread-group dispatch. Pick shape that compiles a single kernel
    // template and is small enough to run quickly in CI without OOM.
    let b: i32 = 1;
    let hk: i32 = 4;
    let hv: i32 = 4;
    let dk: i32 = 128;
    let dv: i32 = 64;

    println!("--- D5d: native gated_delta_step kernel forward equivalence ---");
    println!("shape       = B={b} Hk={hk} Hv={hv} Dk={dk} Dv={dv}");
    println!("n_list      = {n_list:?}");
    println!("n_max       = {n_max}");

    // ── Build per-position random inputs ────────────────────────────────────
    // Seed each tensor independently so values across positions/tensors are
    // uncorrelated. Use normal(0,1) for q/k/v (typical post-norm scale ≈ unit
    // variance) and uniform(0,1) for g/β (the Python kernel feeds sigmoid-like
    // scalars in this range).
    let key0 = mlx_rs::random::key(0xD5D_BEEF).context("random::key failed")?;
    // Split into 5 sub-keys per position; use distinct seeds per position too.
    let n_max_i32 = n_max as i32;

    let mut q_per_pos: Vec<Array> = Vec::with_capacity(n_max);
    let mut k_per_pos: Vec<Array> = Vec::with_capacity(n_max);
    let mut v_per_pos: Vec<Array> = Vec::with_capacity(n_max);
    let mut g_per_pos: Vec<Array> = Vec::with_capacity(n_max);
    let mut beta_per_pos: Vec<Array> = Vec::with_capacity(n_max);

    for pos in 0..n_max as u64 {
        let pk = mlx_rs::random::key(0xD5D_BEEF ^ (pos.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .context("random::key per-pos failed")?;
        let (k1, rest) = mlx_rs::random::split(&pk, 2)?;
        let (k2, rest) = mlx_rs::random::split(&rest, 2)?;
        let (k3, rest) = mlx_rs::random::split(&rest, 2)?;
        let (k4, k5) = mlx_rs::random::split(&rest, 2)?;
        let q = mlx_rs::random::normal::<f32>(&[b, 1, hk, dk][..], None, None, &k1)?;
        let kk = mlx_rs::random::normal::<f32>(&[b, 1, hk, dk][..], None, None, &k2)?;
        let v = mlx_rs::random::normal::<f32>(&[b, 1, hv, dv][..], None, None, &k3)?;
        let g = mlx_rs::random::uniform::<_, f32>(0.0_f32, 1.0_f32, &[b, 1, hv][..], &k4)?;
        let beta = mlx_rs::random::uniform::<_, f32>(0.0_f32, 1.0_f32, &[b, 1, hv][..], &k5)?;
        // Force materialization so each per-position tensor is a stable
        // independent value (no graph capture coupling between paths).
        q.eval()?;
        kk.eval()?;
        v.eval()?;
        g.eval()?;
        beta.eval()?;
        q_per_pos.push(q);
        k_per_pos.push(kk);
        v_per_pos.push(v);
        g_per_pos.push(g);
        beta_per_pos.push(beta);
    }
    let _ = n_max_i32; // silence if shape was used directly above
    drop(key0);

    // ── Sweep ────────────────────────────────────────────────────────────────
    let mut overall_safe = true;
    for &n in &n_list {
        // ─── Path A: single S=N kernel call ────────
        let q_a_refs: Vec<&Array> = q_per_pos.iter().take(n).collect();
        let k_a_refs: Vec<&Array> = k_per_pos.iter().take(n).collect();
        let v_a_refs: Vec<&Array> = v_per_pos.iter().take(n).collect();
        let g_a_refs: Vec<&Array> = g_per_pos.iter().take(n).collect();
        let beta_a_refs: Vec<&Array> = beta_per_pos.iter().take(n).collect();

        let q_a = ops::concatenate_axis(q_a_refs.as_slice(), 1)?;
        let k_a = ops::concatenate_axis(k_a_refs.as_slice(), 1)?;
        let v_a = ops::concatenate_axis(v_a_refs.as_slice(), 1)?;
        let g_a = ops::concatenate_axis(g_a_refs.as_slice(), 1)?;
        let beta_a = ops::concatenate_axis(beta_a_refs.as_slice(), 1)?;

        let state_zero_a = ops::zeros::<f32>(&[b, hv, dv, dk])?;
        let (y_a, state_a) =
            gated_delta_step_kernel(&q_a, &k_a, &v_a, &g_a, &beta_a, &state_zero_a)?;
        y_a.eval()?;
        state_a.eval()?;

        // ─── Path B: N × S=1 kernel calls, chained state ────────
        let mut state_b = ops::zeros::<f32>(&[b, hv, dv, dk])?;
        state_b.eval()?;
        let mut y_rows: Vec<Array> = Vec::with_capacity(n);
        for i in 0..n {
            let (y_i, new_state) = gated_delta_step_kernel(
                &q_per_pos[i],
                &k_per_pos[i],
                &v_per_pos[i],
                &g_per_pos[i],
                &beta_per_pos[i],
                &state_b,
            )?;
            // Force materialization of state_b so subsequent calls see a
            // concrete tensor (not a deferred graph node referencing the
            // original `state_b` shape).
            new_state.eval()?;
            y_i.eval()?;
            y_rows.push(y_i);
            state_b = new_state;
        }
        let y_rows_refs: Vec<&Array> = y_rows.iter().collect();
        let y_b = ops::concatenate_axis(y_rows_refs.as_slice(), 1)?;
        y_b.eval()?;

        // ─── Compare ────────
        let y_diff = ops::abs(&ops::subtract(&y_a, &y_b)?)?;
        let y_max_diff: f32 = y_diff.max(None)?.item::<f32>();
        let state_diff = ops::abs(&ops::subtract(&state_a, &state_b)?)?;
        let state_max_diff: f32 = state_diff.max(None)?.item::<f32>();

        // BAR thresholds. fp32 noise from non-associative reductions accumulates
        // ~O(N · ε); 1e-3 is a generous cap for N ≤ 32 and Dk=128, Dv=64.
        let y_ok = y_max_diff < 1.0e-3;
        let state_ok = state_max_diff < 1.0e-3;
        let bit_id_y = y_max_diff == 0.0;
        let bit_id_state = state_max_diff == 0.0;

        if !(y_ok && state_ok) {
            overall_safe = false;
        }

        println!(
            "N={n:2}: y_max_diff={y_max_diff:.3e} {} state_max_diff={state_max_diff:.3e} {}{}{}",
            if y_ok { "✓" } else { "✗" },
            if state_ok { "✓" } else { "✗" },
            if bit_id_y { " (y bit-id)" } else { "" },
            if bit_id_state { " (state bit-id)" } else { "" },
        );
    }

    println!("\n=== verdict ===");
    println!("  BAR.y     : max|y_A − y_B| < 1e-3 across all N");
    println!("  BAR.state : max|state_A − state_B| < 1e-3 across all N");
    if overall_safe {
        println!("  → kernel sequential-equivalence holds (all N PASS).");
        println!("  → D5a (DFlash native port) is greenlit on the GDN replay axis.");
        Ok(())
    } else {
        println!("  → kernel is NOT sequential-equivalent at the mlx-rs layer.");
        println!("  → D5a port would rediscover the same drift; alternative path =");
        println!("    fork/patch mlx-rs gated_delta_step_kernel dispatch.");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!(
        "bench_native_gdn_equivalence requires --features mlx-native (or mlx-native-metal). \
         Re-run with `cargo run -p lumen-mlx --features mlx-native-metal --example bench_native_gdn_equivalence`."
    );
    std::process::exit(2);
}
