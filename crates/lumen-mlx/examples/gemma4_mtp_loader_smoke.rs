//! the Gemma 4 MTP drafter loader.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/Documents/GitHub/mlx \
//!   DRAFTER_DIR=/path/to/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_mtp_loader_smoke --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::mtp::load_drafter;

    let dir = std::env::var("DRAFTER_DIR")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26B-A4B-it-assistant-bf16".into());
    eprintln!("[mtp-loader-smoke] loading {dir}");

    let drafter = load_drafter(Path::new(&dir)).context("load_drafter failed")?;

    let cfg = &drafter.config;
    let tc = &cfg.text_config;
    println!("=== Gemma 4 MTP drafter loaded ===");
    println!("  backbone_hidden_size = {}", cfg.backbone_hidden_size);
    println!("  drafter hidden_size  = {}", tc.hidden_size);
    println!("  intermediate_size    = {}", tc.intermediate_size);
    println!("  num_hidden_layers    = {}", tc.num_hidden_layers);
    println!("  layer_types          = {:?}", tc.layer_types);
    println!(
        "  attention            = (Q={}, KV={}, head_dim={})",
        tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    );
    println!(
        "  global attention     = (KV={}, head_dim={})",
        tc.num_global_key_value_heads, tc.global_head_dim
    );
    println!("  sliding_window       = {}", tc.sliding_window);
    println!("  vocab_size           = {}", tc.vocab_size);
    println!("  rms_norm_eps         = {}", tc.rms_norm_eps);
    println!("  attention_k_eq_v     = {}", tc.attention_k_eq_v);
    println!("  num_kv_shared_layers = {}", tc.num_kv_shared_layers);
    println!("  tie_word_embeddings  = {}", cfg.tie_word_embeddings);

    println!(
        "\n  embed_tokens   shape = {:?}",
        drafter.embed_tokens.shape()
    );
    println!(
        "  pre_projection shape = {:?}",
        drafter.pre_projection.shape()
    );
    println!(
        "  post_projection shape = {:?}",
        drafter.post_projection.shape()
    );
    println!("  final_norm     shape = {:?}", drafter.final_norm.shape());

    for (i, lw) in drafter.layers.iter().enumerate() {
        println!("\n  Layer {i} ({:?}):", lw.kind);
        println!("    q_proj  = {:?}", lw.attn.q_proj.shape());
        println!("    o_proj  = {:?}", lw.attn.o_proj.shape());
        println!("    q_norm  = {:?}", lw.attn.q_norm.shape());
        println!("    gate    = {:?}", lw.mlp.gate_proj.shape());
        println!("    up      = {:?}", lw.mlp.up_proj.shape());
        println!("    down    = {:?}", lw.mlp.down_proj.shape());
        println!("    input_ln = {:?}", lw.input_layernorm.shape());
        println!("    layer_scalar = {:?}", lw.layer_scalar.shape());
    }

    println!("\n=== Phase 1 loader: PASS ===");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
