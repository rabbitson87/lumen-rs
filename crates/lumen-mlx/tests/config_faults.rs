//! Fault sweep over `config.json` parsing (005 Phase 3).
//!
//! `NativeModelConfig::load` is the first thing that touches a downloaded
//! checkpoint, so it meets every shape a real repo can be in: a partial
//! download, a hand-edited quantization block, a field the upstream trainer
//! emitted as `null`, a `model_type` from a model newer than this build. The
//! invariant is a typed `Err` naming the file — never a panic, and never a
//! silent parse into a config whose numbers are wrong.
//!
//! The alphabet is drawn from what has actually happened. `NullValue` is the
//! JGOS-31B incident: a dense checkpoint carried `num_experts: null`, and
//! `#[serde(default)]` does **not** cover an explicit null (it covers a
//! *missing* key), so the load hard-failed before any migration could run.
//! That was fixed for Gemma 4; this sweep asks the same question of the
//! Qwen 3.5/3.6 parser and pins whatever the answer is.

use lumen_mlx::qwen35_config::NativeModelConfig;
use lumen_testkit::faults::{ConfigMutation, mutate_config};
use serde_json::{Value, json};

/// A minimal but *complete* dense Qwen3.5 config — every required field
/// present, MoE fields absent (which is the dense shape, and the shape the
/// null-vs-missing distinction matters for).
fn dense_config() -> Value {
    json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 4096,
            "head_dim": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_hidden_layers": 4,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "intermediate_size": 8192,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        }
    })
}

fn load_str(text: &str) -> anyhow::Result<NativeModelConfig> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, text).expect("write config");
    NativeModelConfig::load(&path)
}

#[test]
fn the_baseline_config_loads() {
    let cfg = load_str(&serde_json::to_string_pretty(&dense_config()).unwrap())
        .expect("the unmutated fixture must load — otherwise the sweep tests nothing");
    assert_eq!(cfg.text_config.num_hidden_layers, 4);
    assert_eq!(cfg.text_config.hidden_size, 4096);
}

/// Every mutation must fail cleanly or load correctly — never panic, and never
/// load into a config carrying the corrupt value. After each one the pristine
/// config must still load, which is the post-failure integrity check.
#[test]
fn every_config_mutation_is_handled_and_leaves_the_loader_usable() {
    let base = dense_config();
    let mutations = [
        ConfigMutation::RemoveKey("text_config.hidden_size"),
        ConfigMutation::RemoveKey("text_config.layer_types"),
        ConfigMutation::RemoveKey("model_type"),
        ConfigMutation::StringWhereNumber("text_config.num_hidden_layers"),
        ConfigMutation::NegativeNumber("text_config.hidden_size"),
        ConfigMutation::NegativeNumber("text_config.num_attention_heads"),
        ConfigMutation::UnknownModelType,
        ConfigMutation::TruncateJson(50),
        ConfigMutation::TruncateJson(200),
    ];

    for m in &mutations {
        let text = mutate_config(&base, m);
        // The call returning at all is half the assertion.
        let res = load_str(&text);
        match (&m, &res) {
            // An unknown model_type parses fine — rejection is `validate_*`'s
            // job, not the parser's, and conflating them would break every
            // forward-compatible config.
            (ConfigMutation::UnknownModelType, Ok(_)) => {}
            (ConfigMutation::UnknownModelType, Err(e)) => {
                panic!("unknown model_type should parse, not error: {e:#}")
            }
            // A negative number into a `usize` field must be rejected, not
            // wrapped around into a huge dimension.
            (ConfigMutation::NegativeNumber(path), r) => {
                assert!(r.is_err(), "negative {path} parsed as valid");
            }
            (_, r) => {
                assert!(r.is_err(), "mutation {m:?} parsed as valid");
                let msg = format!("{:#}", r.as_ref().unwrap_err());
                assert!(
                    msg.contains("config.json"),
                    "error must name the file so a bad checkpoint is diagnosable: {msg}"
                );
            }
        }
        assert!(
            load_str(&serde_json::to_string_pretty(&base).unwrap()).is_ok(),
            "loader broke after mutation {m:?}"
        );
    }
}

/// The JGOS-31B shape, asked of the Qwen 3.5/3.6 parser.
///
/// `#[serde(default)]` covers a *missing* key, not an explicit `null`. A dense
/// checkpoint that spells its absent MoE fields as `null` — which is what the
/// JGOS-31B config did — therefore hard-fails a `usize` field unless the
/// parser is null-tolerant. This pins the current behavior either way: if it
/// loads, the tolerance is real and must not regress; if it errors, the error
/// must at least name the offending field so the next person does not spend
/// the afternoon the last one did.
#[test]
fn explicit_null_moe_fields_on_a_dense_config() {
    let mut base = dense_config();
    for field in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ] {
        base["text_config"]
            .as_object_mut()
            .unwrap()
            .insert(field.into(), Value::Null);
    }
    let text = serde_json::to_string_pretty(&base).unwrap();

    match load_str(&text) {
        Ok(cfg) => {
            // Null-tolerant: nulls must land on the zero default, not garbage.
            assert_eq!(cfg.text_config.num_experts, 0);
            assert_eq!(cfg.text_config.num_experts_per_tok, 0);
            assert_eq!(cfg.text_config.moe_intermediate_size, 0);
        }
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("num_experts")
                    || msg.contains("num_experts_per_tok")
                    || msg.contains("moe_intermediate_size"),
                "an explicit-null field must be named in the error — this is the JGOS-31B \
                 diagnosis problem, and an unnamed serde error costs an afternoon: {msg}"
            );
        }
    }
}
