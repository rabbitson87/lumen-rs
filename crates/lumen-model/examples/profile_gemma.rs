//! Profile Gemma 4 E4B inference to find bottlenecks.
use std::time::Instant;
use lumen_model::gemma::GemmaModel;

fn main() {
    let model_id = std::env::var("MODEL_ID").unwrap_or("google/gemma-4-E4B-it".into());
    let mut model = GemmaModel::load(&model_id).expect("load failed");

    // Warmup
    let input_ids: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107];
    eprintln!("Warmup...");
    let _ = model.generate(&input_ids, 2, 0.0, 1.0);

    // Benchmark: prefill 10 tokens
    eprintln!("\n=== Prefill (10 tokens) ===");
    let t = Instant::now();
    let _ = model.generate(&input_ids, 1, 0.0, 1.0);
    eprintln!("  Prefill: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    // Benchmark: decode 16 tokens
    eprintln!("\n=== Decode (16 tokens) ===");
    let t = Instant::now();
    let output = model.generate(&input_ids, 16, 0.0, 1.0).unwrap();
    let elapsed = t.elapsed();
    let decode_tokens = output.len().saturating_sub(1); // first token is from prefill
    eprintln!("  Total: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    eprintln!(
        "  Per token: {:.1}ms",
        elapsed.as_secs_f64() * 1000.0 / output.len() as f64
    );
    eprintln!(
        "  Speed: {:.1} tok/s",
        output.len() as f64 / elapsed.as_secs_f64()
    );
    eprintln!("  Output: {:?}", output);
}
