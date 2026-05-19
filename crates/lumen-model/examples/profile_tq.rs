//! Compare: baseline vs TurboQuant speed on actual generation.
use lumen_model::gemma::GemmaModel;
use std::time::Instant;

fn main() {
    let model_id = std::env::var("MODEL_ID").unwrap_or("google/gemma-4-E4B-it".into());

    let input_ids: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107];
    let n_tokens = 16;

    // === Baseline ===
    eprintln!("=== Baseline (no TQ) ===");
    let mut model = GemmaModel::load(&model_id).expect("load failed");
    // warmup
    let _ = model.generate(&input_ids, 2, 0.0, 1.0);

    let t = Instant::now();
    let baseline = model.generate(&input_ids, n_tokens, 0.0, 1.0).unwrap();
    let baseline_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  {n_tokens} tokens in {baseline_ms:.0}ms = {:.1} tok/s",
        baseline.len() as f64 / (baseline_ms / 1000.0)
    );

    // === TurboQuant 4-bit (GPU) ===
    eprintln!("\n=== TurboQuant 4-bit (GPU) ===");
    let mut model = GemmaModel::load(&model_id).expect("load failed");
    let tc = model.text_config();
    model.enable_turboquant(4, tc.num_hidden_layers, tc.num_key_value_heads, tc.head_dim);
    // warmup
    let _ = model.generate(&input_ids, 2, 0.0, 1.0);

    let t = Instant::now();
    let tq_output = model.generate(&input_ids, n_tokens, 0.0, 1.0).unwrap();
    let tq_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  {n_tokens} tokens in {tq_ms:.0}ms = {:.1} tok/s",
        tq_output.len() as f64 / (tq_ms / 1000.0)
    );

    // === Summary ===
    eprintln!("\n=== Comparison ===");
    eprintln!(
        "  Baseline:   {:.1} tok/s",
        baseline.len() as f64 / (baseline_ms / 1000.0)
    );
    eprintln!(
        "  TQ 4-bit:   {:.1} tok/s",
        tq_output.len() as f64 / (tq_ms / 1000.0)
    );
    eprintln!("  Overhead:   {:.0}%", (tq_ms / baseline_ms - 1.0) * 100.0);
}
