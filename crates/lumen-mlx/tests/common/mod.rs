//! `config.json` builders the config contract tests share.
//!
//! Every file under `tests/` is its own crate and uses only some of these,
//! so the rest is dead code in that crate.
#![allow(dead_code)]

use serde_json::{Value, json};
use std::path::Path;

/// Parse `v` the way the loader reads a checkpoint: from a `config.json` on
/// disk, through `load`.
pub fn load_config<T>(v: &Value, load: fn(&Path) -> anyhow::Result<T>) -> T {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_string(v).unwrap()).expect("write");
    load(&path).expect("fixture must parse")
}

/// Set `text_config.<key>`.
pub fn tc(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .insert(key.into(), val);
    v
}

/// Set a top-level key.
pub fn top(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v.as_object_mut().unwrap().insert(key.into(), val);
    v
}

pub fn quant(mode: &str, bits: u32, group: u32) -> Value {
    json!({ "mode": mode, "bits": bits, "group_size": group })
}

pub mod gemma4 {
    use super::{load_config, tc};
    use lumen_mlx::gemma4_config::NativeGemma4Config;
    use serde_json::{Value, json};

    /// A minimal **dense** Gemma 4 config that validates. `sliding_window_pattern`
    /// defaults such that 4 layers with one full-attention entry is consistent.
    pub fn dense() -> Value {
        json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 5376,
                "head_dim": 256,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "intermediate_size": 8192,
                "rms_norm_eps": 1e-6,
                "sliding_window": 512,
                "sliding_window_pattern": 4,
                "max_position_embeddings": 4096,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
                ],
                "rope_parameters": {
                    "full_attention": { "rope_theta": 1000000.0 },
                    "sliding_attention": { "rope_theta": 10000.0 }
                }
            }
        })
    }

    /// The 26B-A4B shape: MoE enabled, so the expert-size rules apply.
    pub fn moe() -> Value {
        let v = dense();
        let v = tc(&v, "enable_moe_block", json!(true));
        let v = tc(&v, "num_experts", json!(128));
        let v = tc(&v, "top_k_experts", json!(8));
        tc(&v, "moe_intermediate_size", json!(1024))
    }

    pub fn load(v: &Value) -> NativeGemma4Config {
        load_config(v, NativeGemma4Config::load)
    }
}
