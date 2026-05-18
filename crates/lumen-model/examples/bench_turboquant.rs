//! TurboQuant KV Cache Compression Benchmark
//!
//! Compares baseline (uncompressed) vs TurboQuant GPU (compressed) generation
//! on actual Gemma 4 E4B-it model.
//!
//! Usage:
//!   MODEL_ID="google/gemma-4-E4B-it" cargo run --example bench_turboquant -p lumen-model --release

use std::time::Instant;

use lumen_model::gemma::GemmaModel;

fn main() {
    eprintln!("=== TurboQuant KV Cache Compression Benchmark ===\n");

    // ── Step 1: Load model ──────────────────────────────────────────────
    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "google/gemma-4-E4B-it".to_string());
    let mut model = GemmaModel::load(&model_id).expect("failed to load model");

    let head_dim_sliding = model.text_config().head_dim;
    let head_dim_global = model
        .text_config()
        .global_head_dim
        .unwrap_or(head_dim_sliding);
    let n_kv_heads = model.text_config().num_key_value_heads;
    let n_layers = model.text_config().num_hidden_layers;

    // ── Step 2: Build prompt ────────────────────────────────────────────
    let prompt_text = "What is the capital of France? Explain in detail.";
    let mut input_ids = vec![2u32, 105]; // <bos>, <|turn>
    input_ids.extend(model.encode("user\n").unwrap());
    input_ids.extend(model.encode(prompt_text).unwrap());
    input_ids.push(106); // <turn|>
    input_ids.extend(model.encode("\n").unwrap());
    input_ids.push(105); // <|turn>
    input_ids.extend(model.encode("model\n").unwrap());

    let prompt_len = input_ids.len();
    eprintln!("Prompt: \"{prompt_text}\"");
    eprintln!("Prompt tokens: {prompt_len}");
    eprintln!(
        "Config: layers={n_layers}, kv_heads={n_kv_heads}, sliding_dim={head_dim_sliding}, global_dim={head_dim_global}\n"
    );

    let max_tokens = 64;

    // ── Step 3: Baseline generation (greedy) ────────────────────────────
    eprintln!("--- Baseline (candle KV cache, BF16) ---");
    let baseline_start = Instant::now();
    let baseline_ids = model
        .generate(&input_ids, max_tokens, 0.0, 1.0)
        .expect("baseline generate failed");
    let baseline_elapsed = baseline_start.elapsed();
    let baseline_text = model.decode(&baseline_ids).unwrap();
    eprintln!("  Time:   {:.2}s", baseline_elapsed.as_secs_f64());
    eprintln!(
        "  Speed:  {:.1} tok/s",
        baseline_ids.len() as f64 / baseline_elapsed.as_secs_f64()
    );
    eprintln!("  Output: \"{baseline_text}\"");
    eprintln!();

    // ── Step 4: TurboQuant 4-bit generation ─────────────────────────────
    eprintln!("--- TurboQuant GPU (4-bit compressed KV) ---");

    // Drop previous model to free ~5GB before loading next
    drop(model);
    let mut model = GemmaModel::load(&model_id).expect("reload failed");
    model.enable_turboquant(4, n_layers, n_kv_heads, head_dim_sliding);

    let tq4_start = Instant::now();
    let tq4_ids = model
        .generate(&input_ids, max_tokens, 0.0, 1.0)
        .expect("TQ 4-bit generate failed");
    let tq4_elapsed = tq4_start.elapsed();
    let tq4_text = model.decode(&tq4_ids).unwrap();
    eprintln!("  Time:   {:.2}s", tq4_elapsed.as_secs_f64());
    eprintln!(
        "  Speed:  {:.1} tok/s",
        tq4_ids.len() as f64 / tq4_elapsed.as_secs_f64()
    );
    eprintln!("  Output: \"{tq4_text}\"");
    eprintln!();

    let matching_4bit = baseline_ids
        .iter()
        .zip(tq4_ids.iter())
        .take_while(|(a, b)| a == b)
        .count();

    // ── Step 5: TurboQuant 3-bit generation ─────────────────────────────
    eprintln!("--- TurboQuant GPU (3-bit compressed KV) ---");

    drop(model);
    let mut model = GemmaModel::load(&model_id).expect("reload failed");
    model.enable_turboquant(3, n_layers, n_kv_heads, head_dim_sliding);

    let tq3_start = Instant::now();
    let tq3_ids = model
        .generate(&input_ids, max_tokens, 0.0, 1.0)
        .expect("TQ 3-bit generate failed");
    let tq3_elapsed = tq3_start.elapsed();
    let tq3_text = model.decode(&tq3_ids).unwrap();
    eprintln!("  Time:   {:.2}s", tq3_elapsed.as_secs_f64());
    eprintln!(
        "  Speed:  {:.1} tok/s",
        tq3_ids.len() as f64 / tq3_elapsed.as_secs_f64()
    );
    eprintln!("  Output: \"{tq3_text}\"");
    eprintln!();

    let matching_3bit = baseline_ids
        .iter()
        .zip(tq3_ids.iter())
        .take_while(|(a, b)| a == b)
        .count();

    // ── Step 6: Memory estimation ───────────────────────────────────────
    let n_sliding = 35usize;
    let n_global = 7usize;

    for &seq_len in &[prompt_len, 512, 2048, 8192] {
        eprintln!("--- Memory Estimation ({seq_len} tokens) ---");

        // Baseline: BF16 K+V cache
        let baseline_sliding = n_sliding * n_kv_heads * seq_len * head_dim_sliding * 2 * 2; // K+V, 2 bytes each
        let baseline_global = n_global * n_kv_heads * seq_len * head_dim_global * 2 * 2;
        let baseline_total = baseline_sliding + baseline_global;

        // TurboQuant 3-bit: per scalar ≈ 3/8 bytes (codes) + scale overhead
        let tq3_bits_per_scalar = 3;
        let tq3_sliding = n_sliding * n_kv_heads * seq_len * head_dim_sliding * tq3_bits_per_scalar / 8 * 2 // K+V
            + n_sliding * n_kv_heads * seq_len * 4 * 2; // scales (f32 per vector, K+V)
        let tq3_global =
            n_global * n_kv_heads * seq_len * head_dim_global * tq3_bits_per_scalar / 8 * 2
                + n_global * n_kv_heads * seq_len * 4 * 2;
        let tq3_total = tq3_sliding + tq3_global;

        let savings_pct = (1.0 - tq3_total as f64 / baseline_total as f64) * 100.0;

        eprintln!("  Baseline (BF16): {:.2} MB", baseline_total as f64 / 1e6,);
        eprintln!(
            "  TurboQuant 3-bit: {:.2} MB  ({savings_pct:.1}% savings)",
            tq3_total as f64 / 1e6,
        );
        eprintln!();
    }

    // ── Summary ─────────────────────────────────────────────────────────
    eprintln!("=== Summary ===");
    eprintln!("Model:             {model_id}");
    eprintln!("Prompt tokens:     {prompt_len}");
    eprintln!("Generated:         {} tokens", max_tokens);
    eprintln!(
        "Baseline:          {:.2}s ({:.1} tok/s)",
        baseline_elapsed.as_secs_f64(),
        baseline_ids.len() as f64 / baseline_elapsed.as_secs_f64()
    );
    eprintln!(
        "TQ 4-bit:          {:.2}s ({:.1} tok/s)",
        tq4_elapsed.as_secs_f64(),
        tq4_ids.len() as f64 / tq4_elapsed.as_secs_f64()
    );
    eprintln!(
        "TQ 3-bit:          {:.2}s ({:.1} tok/s)",
        tq3_elapsed.as_secs_f64(),
        tq3_ids.len() as f64 / tq3_elapsed.as_secs_f64()
    );
    eprintln!(
        "4-bit token match: {matching_4bit}/{} prefix identical",
        baseline_ids.len().min(tq4_ids.len())
    );
    eprintln!(
        "3-bit token match: {matching_3bit}/{} prefix identical",
        baseline_ids.len().min(tq3_ids.len())
    );
    eprintln!();
    eprintln!("Baseline: \"{baseline_text}\"");
    eprintln!("TQ 4-bit: \"{tq4_text}\"");
    eprintln!("TQ 3-bit: \"{tq3_text}\"");
}
