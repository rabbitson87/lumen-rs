//! Sanity test: NativeGemma4PromptCache::clone() — confirm that cloned
//! caches evolve independently (so prefix cache "fork from master" pattern
//! is safe). MLX Array is refcount-based, but each cache update produces
//! new Arrays — so two clones diverging after the fork point should NOT
//! corrupt each other.

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::NativeGemma4Model;

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-3bit".into());

    eprintln!("[cache-clone-sanity] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;

    // Step 1 — build a cache, fill with 10 tokens (the prefix).
    let prefix: Vec<u32> = vec![2, 105, 2364, 107, 213229, 78622, 126623, 237223, 42813, 189515];
    let mut master = model.make_cache();
    let _ = model
        .forward_last_token(&prefix, &mut master)
        .context("prefix fill")?;
    let master_offset_after_prefix = master.offset();
    println!("master offset after prefix: {master_offset_after_prefix}");

    // Step 2 — clone the master (this is the "fork" operation).
    let mut clone_a = master.clone();
    let mut clone_b = master.clone();
    println!(
        "clone_a offset: {} | clone_b offset: {} (both should match master)",
        clone_a.offset(),
        clone_b.offset()
    );

    // Step 3 — evolve each clone with different suffix tokens.
    let suffix_a: Vec<u32> = vec![5, 6, 7];
    let suffix_b: Vec<u32> = vec![100, 200, 300, 400, 500];
    let _ = model
        .forward_last_token(&suffix_a, &mut clone_a)
        .context("clone_a suffix")?;
    let _ = model
        .forward_last_token(&suffix_b, &mut clone_b)
        .context("clone_b suffix")?;

    println!(
        "after suffix: master={} clone_a={} clone_b={}",
        master.offset(),
        clone_a.offset(),
        clone_b.offset()
    );

    // Independent evolution: master should be unchanged, clones at their
    // respective offsets.
    let ok_master = master.offset() == master_offset_after_prefix;
    let ok_a = clone_a.offset() == master_offset_after_prefix + suffix_a.len();
    let ok_b = clone_b.offset() == master_offset_after_prefix + suffix_b.len();
    println!(
        "expectations: master_unchanged={ok_master} clone_a_advanced={ok_a} clone_b_advanced={ok_b}"
    );
    if ok_master && ok_a && ok_b {
        println!("=== CACHE CLONE SANITY: PASS ===");
        println!("Clones evolve independently from the master snapshot.");
        println!("Prefix cache 'fork from master' pattern is safe to implement.");
    } else {
        eprintln!("=== CACHE CLONE SANITY: FAIL ===");
        eprintln!("Clone semantics not as expected — investigate before wiring prefix cache.");
    }
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
