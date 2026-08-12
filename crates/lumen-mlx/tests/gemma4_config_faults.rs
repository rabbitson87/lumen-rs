//! The JGOS-31B null sweep, asked of Gemma 4 (005 Phase 3 follow-up).
//!
//! `config_faults.rs` asked this of the Qwen 3.5/3.6 parser and found a real
//! defect: `#[serde(default)]` covers a *missing* key, not an explicit `null`,
//! so a dense checkpoint spelling its absent MoE fields as `null` hard-failed
//! before any migration could run. That was fixed there with `null_as_default`.
//!
//! The reason to ask it here rather than assume: JGOS-31B is *remembered* as a
//! Gemma 4 fix, but what that work actually changed was weight resolution
//! (grouping the MoE weights behind an `Option`) and family routing — **not**
//! the config parser. A memory of "we fixed that" is not an answer.
//!
//! It was not fixed. Measured before this file's fix: a dense Gemma 4 config
//! carrying `"num_experts": null` failed with `invalid type: null, expected
//! usize at line 31 column 23` — the same shape, in the family it was recorded
//! against, naming neither the field nor the architecture. Now null-tolerant
//! through the shared `config_serde::null_as_default`.
//!
//! The boundary matters as much as the tolerance, so both directions are
//! pinned: architecture-optional fields must treat null as missing, and fields
//! the architecture *requires* must still reject it. Silently defaulting
//! `sliding_window` to 0 would not fail — it would build a model with the wrong
//! attention window, which is the invisible-wrongness class 005 exists for.

#![cfg(feature = "mlx-native")]

use lumen_mlx::gemma4::NativeGemma4Config;
use lumen_testkit::faults::{ConfigMutation, mutate_config};
use serde_json::{Value, json};

/// A minimal but complete **dense** Gemma 4 config: MoE disabled, MoE fields
/// absent. That is the shape the null distinction matters for — a dense
/// checkpoint has nothing to put in those fields and different exporters
/// disagree on whether that means "omit" or "null".
fn dense_config() -> Value {
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
            "max_position_embeddings": 4096,
            "final_logit_softcapping": 0.0,
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

fn load_str(text: &str) -> anyhow::Result<NativeGemma4Config> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, text).expect("write config");
    NativeGemma4Config::load(&path)
}

#[test]
fn the_baseline_config_loads() {
    let cfg = load_str(&serde_json::to_string_pretty(&dense_config()).unwrap())
        .expect("the unmutated fixture must load — otherwise the sweep tests nothing");
    assert_eq!(cfg.text_config.num_hidden_layers, 4);
    assert_eq!(cfg.text_config.hidden_size, 5376);
}

/// The question this file exists to answer.
#[test]
fn explicit_null_moe_fields_on_a_dense_config() {
    let mut base = dense_config();
    for field in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        base["text_config"]
            .as_object_mut()
            .unwrap()
            .insert(field.into(), Value::Null);
    }
    let text = serde_json::to_string_pretty(&base).unwrap();

    match load_str(&text) {
        Ok(cfg) => {
            // Null-tolerant: nulls must land on the zero default, not on a
            // garbage value that only shows up as a wrong expert count later.
            assert_eq!(cfg.text_config.num_experts, 0);
            assert_eq!(cfg.text_config.top_k_experts, 0);
        }
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("num_experts")
                    || msg.contains("top_k_experts")
                    || msg.contains("moe_intermediate_size"),
                "an explicit-null field must be named in the error — this is exactly the \
                 JGOS-31B diagnosis problem, and an unnamed serde error on a config this \
                 wide costs an afternoon: {msg}"
            );
        }
    }
}

/// Null tolerance has a boundary, and getting it wrong in either direction is
/// a bug — so both sides are asserted.
///
/// A field that is legitimately absent on *some* architecture must treat null
/// and missing identically. A field the architecture genuinely requires must
/// **reject** null: silently defaulting `sliding_window` to 0 would not fail,
/// it would build a model with the wrong attention window, which is exactly the
/// invisible-wrongness class this task exists to eliminate.
#[test]
fn null_tolerance_stops_at_the_architecture_boundary() {
    let base = dense_config();

    // Absent on a dense checkpoint → null means "not applicable".
    for field in [
        "num_experts",
        "top_k_experts",
        "moe_intermediate_size",
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "use_double_wide_mlp",
    ] {
        let mut v = base.clone();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .insert(field.into(), Value::Null);
        load_str(&serde_json::to_string_pretty(&v).unwrap())
            .unwrap_or_else(|e| panic!("`{field}: null` must load as the default: {e:#}"));
    }

    // Required by the architecture → null is corruption. Defaulting these
    // would produce a silently mis-shaped model.
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "sliding_window",
        "vocab_size",
        "intermediate_size",
    ] {
        let mut v = base.clone();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .insert(field.into(), Value::Null);
        let Err(e) = load_str(&serde_json::to_string_pretty(&v).unwrap()) else {
            panic!(
                "`{field}: null` loaded — a required dimension silently became 0, which \
                 builds a wrong model rather than failing"
            )
        };
        let msg = format!("{e:#}");
        assert!(
            msg.contains("config.json") && msg.contains("line"),
            "the error must name the file and a position so the bad key is findable: {msg}"
        );
        // Post-failure integrity.
        assert!(
            load_str(&serde_json::to_string_pretty(&base).unwrap()).is_ok(),
            "loader broke after a null in {field}"
        );
    }
}

/// The same structured mutations `config_faults.rs` runs against Qwen. Gemma 4
/// has its own parser, so coverage of one says nothing about the other.
#[test]
fn every_config_mutation_is_handled_and_leaves_the_loader_usable() {
    let base = dense_config();
    let mutations = [
        ConfigMutation::RemoveKey("text_config.hidden_size"),
        ConfigMutation::RemoveKey("model_type"),
        ConfigMutation::StringWhereNumber("text_config.num_hidden_layers"),
        ConfigMutation::NegativeNumber("text_config.hidden_size"),
        ConfigMutation::NegativeNumber("text_config.num_attention_heads"),
        ConfigMutation::UnknownModelType,
        ConfigMutation::TruncateJson(40),
        ConfigMutation::TruncateJson(150),
    ];

    for m in &mutations {
        let res = load_str(&mutate_config(&base, m));
        match (&m, &res) {
            // Forward compatibility: an unrecognized `model_type` must parse;
            // rejecting it is `validate_*`'s job, not the parser's.
            (ConfigMutation::UnknownModelType, Ok(_)) => {}
            (ConfigMutation::UnknownModelType, Err(e)) => {
                panic!("unknown model_type should parse, not error: {e:#}")
            }
            (ConfigMutation::NegativeNumber(path), r) => {
                assert!(
                    r.is_err(),
                    "negative {path} parsed as valid — a wrapped-around usize becomes a \
                     dimension nothing checks again"
                );
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
