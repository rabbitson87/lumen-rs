fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("./models/gemma-4-31B-it-Q4_K_M.gguf".into());
    let mut file = std::fs::File::open(&path).expect("open failed");
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).expect("read failed");

    for key in ct.metadata.keys() {
        if key.contains("head_count")
            || key.contains("key_length")
            || key.contains("value_length")
            || key.contains("sliding")
            || key.contains("block_count")
            || key.contains("k_eq_v")
        {
            let val = &ct.metadata[key];
            println!("{key}: {val:?}");
        }
    }

    // Check tensor names for layer 5
    for name in ct.tensor_infos.keys() {
        if name.contains("blk.5") {
            let info = &ct.tensor_infos[name];
            println!("tensor: {name}  shape={:?}", info.shape);
        }
    }
}
