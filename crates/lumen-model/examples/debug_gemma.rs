use lumen_model::gemma::GemmaModel;

fn main() {
    eprintln!("=== Gemma4 E4B Debug Test ===");
    let mut model = GemmaModel::load("google/gemma-4-E4B-it").expect("failed to load model");

    // Same input as HF reference
    let input_ids: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107];
    eprintln!("Input IDs: {:?}", input_ids);

    let output = model
        .generate(&input_ids, 8, 0.0, 1.0)
        .expect("generate failed");
    eprintln!("Output IDs: {:?}", output);

    let text = model.decode(&output).expect("decode failed");
    eprintln!("Output text: {text}");

    // Expected from HF: [9259, 236888, 2088, 740, 564, 1601, 611, 3124]
    // = "Hello! How can I help you today"
}
