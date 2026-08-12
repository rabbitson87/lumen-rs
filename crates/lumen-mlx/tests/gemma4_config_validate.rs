//! Contract coverage for the Gemma 4 config validators (005 Phase 4.1).
//!
//! Companion to `qwen35_config_validate.rs`, written for the same reason: the
//! fault sweeps drive `load()`, which only parses, while every real rule lives
//! in `validate_*` that the mlx-native model loader calls. The branch baseline
//! put a number on it — 110 branches here, **107 missed, 2.73%**.
//!
//! Gemma 4's validator carries two things Qwen's does not, and both are worth
//! covering precisely:
//!
//! * **A cross-mode quantization rule.** Overrides must match the default's
//!   `group_size` *only when their modes agree*; a mismatch across modes is
//!   legal, because per-tensor dispatch hands each tensor its own
//!   `(group_size, bits, mode)` triple. A test that only checked "mismatch is
//!   rejected" would lock in the opposite of the shipping behaviour.
//! * **Layer-kind accessors** (`head_dim_for`, `n_kv_heads_for`,
//!   `use_k_eq_v_for`) whose whole job is to return something *different* for
//!   full vs sliding layers. Getting one wrong builds attention with the wrong
//!   head count and never errors.

use lumen_mlx::gemma4_config::{NativeGemma4Config, NativeGemma4LayerType};
use serde_json::{Value, json};

// ───────────────────────────── fixtures ─────────────────────────────

/// A minimal **dense** Gemma 4 config that validates. `sliding_window_pattern`
/// defaults such that 4 layers with one full-attention entry is consistent.
fn dense() -> Value {
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
fn moe() -> Value {
    let v = dense();
    let v = tc(&v, "enable_moe_block", json!(true));
    let v = tc(&v, "num_experts", json!(128));
    let v = tc(&v, "top_k_experts", json!(8));
    tc(&v, "moe_intermediate_size", json!(1024))
}

fn load(v: &Value) -> NativeGemma4Config {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_string(v).unwrap()).expect("write");
    NativeGemma4Config::load(&path).expect("fixture must parse")
}

fn tc(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .insert(key.into(), val);
    v
}

fn top(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v.as_object_mut().unwrap().insert(key.into(), val);
    v
}

fn quant(mode: &str, bits: u32, group: u32) -> Value {
    json!({ "mode": mode, "bits": bits, "group_size": group })
}

// ───────────────────────────── baselines ─────────────────────────────

#[test]
fn both_baselines_validate() {
    load(&dense())
        .validate_gemma4_family()
        .expect("the dense fixture must validate — otherwise every mutation below is noise");
    load(&moe())
        .validate_gemma4_family()
        .expect("the MoE fixture must validate");
}

// ───────────────────────── identity / architectures ─────────────────────────

#[test]
fn model_type_and_architectures_are_both_checked() {
    for mt in ["gemma3", "qwen3_5", "", "Gemma4"] {
        let err = load(&top(&dense(), "model_type", json!(mt)))
            .validate_gemma4_family()
            .expect_err("a non-gemma4 model_type must be rejected");
        assert!(format!("{err:#}").contains("model_type"), "got: {err:#}");
    }

    // Absent/empty architectures is fine — many conversions omit the list.
    load(&top(&dense(), "architectures", json!([])))
        .validate_gemma4_family()
        .expect("an empty architectures list must not be a failure");

    // Present but wrong.
    let err = load(&top(
        &dense(),
        "architectures",
        json!(["Gemma3ForCausalLM"]),
    ))
    .validate_gemma4_family()
    .expect_err("a non-Gemma4 architecture must be rejected");
    assert!(format!("{err:#}").contains("architectures"), "got: {err:#}");

    // Present and right, even alongside others.
    load(&top(
        &dense(),
        "architectures",
        json!(["Something", "Gemma4ForConditionalGeneration"]),
    ))
    .validate_gemma4_family()
    .expect("the expected architecture anywhere in the list is enough");
}

// ───────────────────────────── core dims ─────────────────────────────

#[test]
fn every_core_dim_is_rejected_at_zero_individually() {
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "intermediate_size",
    ] {
        let mut v = tc(&dense(), field, json!(0));
        if field == "num_hidden_layers" {
            v = tc(&v, "layer_types", json!([]));
        }
        let err = load(&v).validate_gemma4_family().unwrap_err().to_string();
        assert!(
            err.contains("zero-valued core dims"),
            "{field}=0 gave: {err}"
        );
    }
}

#[test]
fn layer_types_length_must_match_the_layer_count() {
    for types in [
        json!(["sliding_attention", "full_attention"]),
        json!([
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention"
        ]),
    ] {
        let err = load(&tc(&dense(), "layer_types", types))
            .validate_gemma4_family()
            .expect_err("mismatched layer_types must be rejected");
        assert!(format!("{err:#}").contains("layer_types length"), "{err:#}");
    }
}

/// Both of these default to something usable, so a config that sets them to a
/// degenerate value did so deliberately — and a 0-token attention window or a
/// non-positive softcap is a silently wrong model, not a crash.
#[test]
fn sliding_window_and_softcapping_must_be_positive() {
    let err = load(&tc(&dense(), "sliding_window", json!(0)))
        .validate_gemma4_family()
        .expect_err("a zero attention window must be rejected");
    assert!(format!("{err:#}").contains("sliding_window"), "{err:#}");

    for cap in [json!(0.0), json!(-1.0), json!(-30.0)] {
        let err = load(&tc(&dense(), "final_logit_softcapping", cap.clone()))
            .validate_gemma4_family()
            .expect_err("a non-positive softcap must be rejected");
        assert!(
            format!("{err:#}").contains("final_logit_softcapping"),
            "cap={cap}: {err:#}"
        );
    }
}

/// The full/sliding split has to contain at least one full-attention layer:
/// with none, nothing in the model ever attends beyond the window.
#[test]
fn the_layer_split_must_contain_a_full_attention_layer() {
    let all_sliding = json!([
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention"
    ]);
    let err = load(&tc(&dense(), "layer_types", all_sliding))
        .validate_gemma4_family()
        .expect_err("an all-sliding stack must be rejected");
    assert!(format!("{err:#}").contains("full_attention"), "{err:#}");

    // All-full is allowed: the count is bounded above by num_hidden_layers,
    // and a model that attends globally everywhere is expensive, not wrong.
    let all_full = json!([
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention"
    ]);
    load(&tc(&dense(), "layer_types", all_full))
        .validate_gemma4_family()
        .expect("an all-full stack is expensive, not invalid");
}

// ───────────────────────────── MoE arm ─────────────────────────────

#[test]
fn the_moe_arm_is_only_checked_when_it_is_enabled() {
    // Disabled: the expert fields are ignored even when degenerate. This is
    // the dense-checkpoint shape and must not be a failure.
    let v = tc(
        &tc(&dense(), "num_experts", json!(0)),
        "top_k_experts",
        json!(0),
    );
    load(&v)
        .validate_gemma4_family()
        .expect("a dense config must not be judged by MoE rules");

    // Enabled: each rule fires on its own.
    for (field, val, needle) in [
        ("num_experts", json!(0), "num_experts=0"),
        ("top_k_experts", json!(0), "top_k_experts"),
        ("top_k_experts", json!(129), "top_k_experts"),
        ("moe_intermediate_size", json!(0), "moe_intermediate_size"),
    ] {
        let err = load(&tc(&moe(), field, val.clone()))
            .validate_gemma4_family()
            .unwrap_err();
        assert!(
            format!("{err:#}").contains(needle),
            "{field}={val} should mention {needle:?}: {err:#}"
        );
    }

    // top_k == num_experts is the legal boundary.
    load(&tc(&moe(), "top_k_experts", json!(128)))
        .validate_gemma4_family()
        .expect("top_k == num_experts is a boundary, not an error");
}

// ───────────────────────────── quantization ─────────────────────────────

#[test]
fn the_quantization_block_is_read_from_either_key() {
    // `quantization` wins over `quantization_config` when both are present —
    // pinned because the precedence is silent otherwise.
    let both = top(
        &top(&dense(), "quantization", quant("affine", 4, 64)),
        "quantization_config",
        quant("mxfp4", 4, 32),
    );
    let cfg = load(&both);
    let eff = cfg.effective_quantization().expect("one of the two");
    assert_eq!(eff.mode, "affine");
    assert_eq!(eff.group_size, 64);

    // Only the fallback key.
    let cfg = load(&top(&dense(), "quantization_config", quant("mxfp4", 4, 32)));
    assert_eq!(cfg.effective_quantization().unwrap().mode, "mxfp4");

    // Neither: an unquantized checkpoint.
    assert!(load(&dense()).effective_quantization().is_none());
    load(&dense())
        .validate_gemma4_family()
        .expect("bf16 configs carry no quantization block");
}

#[test]
fn accepted_and_rejected_quant_recipes() {
    for mode in ["affine", "mxfp4", "mxfp8", "nvfp4"] {
        load(&top(&dense(), "quantization", quant(mode, 4, 64)))
            .validate_gemma4_family()
            .unwrap_or_else(|e| panic!("{mode} is a supported mode: {e:#}"));
    }
    for bits in [2u32, 3, 4, 5, 6, 8] {
        load(&top(&dense(), "quantization", quant("affine", bits, 64)))
            .validate_gemma4_family()
            .unwrap_or_else(|e| panic!("{bits}-bit is mlx-supported: {e:#}"));
    }
    for bits in [1u32, 7, 9, 16, 32] {
        let err = load(&top(&dense(), "quantization", quant("affine", bits, 64)))
            .validate_gemma4_family()
            .expect_err("an mlx-unsupported bit width must be rejected");
        assert!(format!("{err:#}").contains("bits"), "{bits}: {err:#}");
    }
    for mode in ["int4", "gptq", ""] {
        assert!(
            load(&top(&dense(), "quantization", quant(mode, 4, 64)))
                .validate_gemma4_family()
                .is_err(),
            "mode={mode:?} must be rejected"
        );
    }
    assert!(
        load(&top(&dense(), "quantization", quant("affine", 4, 0)))
            .validate_gemma4_family()
            .is_err(),
        "group_size 0 divides by zero downstream"
    );
}

/// The cross-mode override rule, in **both** directions.
///
/// This is the subtle one. A group_size mismatch is rejected when the override
/// shares the default's mode, and allowed when it does not — because per-tensor
/// dispatch gives each tensor its own `(group_size, bits, mode)`. A test that
/// asserted only the rejection would enshrine the opposite of what ships
/// (MXFP4 g=32 default with AFFINE g=64 embedding overrides).
#[test]
fn override_group_size_is_only_constrained_within_one_mode() {
    // Same mode, mismatched group → rejected.
    let mut q = quant("affine", 4, 64);
    q.as_object_mut().unwrap().insert(
        "model.embed_tokens".into(),
        json!({ "group_size": 32, "bits": 4, "mode": "affine" }),
    );
    let err = load(&top(&dense(), "quantization", q))
        .validate_gemma4_family()
        .expect_err("same-mode group mismatch must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("model.embed_tokens") && msg.contains("group_size"),
        "{msg}"
    );

    // Same mode, matching group → fine.
    let mut q = quant("affine", 4, 64);
    q.as_object_mut().unwrap().insert(
        "model.embed_tokens".into(),
        json!({ "group_size": 64, "bits": 8, "mode": "affine" }),
    );
    load(&top(&dense(), "quantization", q))
        .validate_gemma4_family()
        .expect("same mode, same group, different bits is a normal mixed recipe");

    // DIFFERENT mode, mismatched group → allowed. The shipping shape.
    let mut q = quant("mxfp4", 4, 32);
    q.as_object_mut().unwrap().insert(
        "model.embed_tokens".into(),
        json!({ "group_size": 64, "bits": 8, "mode": "affine" }),
    );
    load(&top(&dense(), "quantization", q))
        .validate_gemma4_family()
        .expect("cross-mode group_size mismatch is how MXFP4 models carry AFFINE overrides");

    // An override with no `mode` inherits affine — so against an affine
    // default it IS same-mode and the group rule applies.
    let mut q = quant("affine", 4, 64);
    q.as_object_mut().unwrap().insert(
        "model.embed_tokens".into(),
        json!({ "group_size": 32, "bits": 4 }),
    );
    assert!(
        load(&top(&dense(), "quantization", q))
            .validate_gemma4_family()
            .is_err(),
        "a mode-less override inherits affine, so it must obey the same-mode group rule"
    );

    // ...and against an mxfp4 default the same mode-less override is cross-mode.
    let mut q = quant("mxfp4", 4, 32);
    q.as_object_mut().unwrap().insert(
        "model.embed_tokens".into(),
        json!({ "group_size": 64, "bits": 4 }),
    );
    load(&top(&dense(), "quantization", q))
        .validate_gemma4_family()
        .expect("a mode-less override against a non-affine default is cross-mode");
}

#[test]
fn override_bits_are_checked_too() {
    for bits in [1u32, 7, 9] {
        let mut q = quant("affine", 4, 64);
        q.as_object_mut().unwrap().insert(
            "model.embed_tokens".into(),
            json!({ "group_size": 64, "bits": bits, "mode": "affine" }),
        );
        let err = load(&top(&dense(), "quantization", q))
            .validate_gemma4_family()
            .expect_err("an unsupported override bit width must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model.embed_tokens") && msg.contains("bits"),
            "{bits}: {msg}"
        );
    }
}

// ───────────────────────── layer-kind accessors ─────────────────────────

/// These return a *different* answer per layer kind, which is precisely why a
/// wrong one is invisible: attention still runs, with the wrong shape.
#[test]
fn layer_kind_accessors_resolve_per_kind() {
    use NativeGemma4LayerType::{FullAttention, SlidingAttention};

    // Without a global override, both kinds share head_dim.
    let cfg = load(&dense());
    assert_eq!(cfg.text_config.head_dim_for(FullAttention), 256);
    assert_eq!(cfg.text_config.head_dim_for(SlidingAttention), 256);

    // With one, only full attention picks it up.
    let cfg = load(&tc(&dense(), "global_head_dim", json!(128)));
    assert_eq!(cfg.text_config.head_dim_for(FullAttention), 128);
    assert_eq!(cfg.text_config.head_dim_for(SlidingAttention), 256);

    // k_eq_v applies to full attention only, and only when enabled.
    let cfg = load(&dense());
    assert!(!cfg.text_config.use_k_eq_v_for(FullAttention));
    let cfg = load(&tc(&dense(), "attention_k_eq_v", json!(true)));
    assert!(cfg.text_config.use_k_eq_v_for(FullAttention));
    assert!(!cfg.text_config.use_k_eq_v_for(SlidingAttention));

    // The global KV-head override needs BOTH k_eq_v and the field present.
    let cfg = load(&tc(&dense(), "num_global_key_value_heads", json!(2)));
    assert_eq!(
        cfg.text_config.n_kv_heads_for(FullAttention),
        4,
        "without attention_k_eq_v the global count must be ignored"
    );
    let v = tc(
        &tc(&dense(), "attention_k_eq_v", json!(true)),
        "num_global_key_value_heads",
        json!(2),
    );
    let cfg = load(&v);
    assert_eq!(cfg.text_config.n_kv_heads_for(FullAttention), 2);
    assert_eq!(cfg.text_config.n_kv_heads_for(SlidingAttention), 4);

    // k_eq_v on but no global count → fall back.
    let cfg = load(&tc(&dense(), "attention_k_eq_v", json!(true)));
    assert_eq!(cfg.text_config.n_kv_heads_for(FullAttention), 4);

    // RoPE resolves to the per-kind block.
    let cfg = load(&dense());
    assert_eq!(
        cfg.text_config.rope_for(FullAttention).rope_theta,
        1_000_000.0
    );
    assert_eq!(
        cfg.text_config.rope_for(SlidingAttention).rope_theta,
        10_000.0
    );
}

#[test]
fn the_layer_kind_predicates_are_mutually_exclusive() {
    use NativeGemma4LayerType::{FullAttention, SlidingAttention};
    for kind in [FullAttention, SlidingAttention] {
        assert_ne!(
            kind.is_full(),
            kind.is_sliding(),
            "{kind:?} must be exactly one of full/sliding"
        );
    }
    assert!(FullAttention.is_full());
    assert!(SlidingAttention.is_sliding());
}

/// Same scalar-or-list tolerance as Qwen's; exporters disagree.
#[test]
fn eos_token_id_accepts_both_a_scalar_and_a_list() {
    assert_eq!(
        load(&top(&dense(), "eos_token_id", json!(106))).eos_token_ids,
        vec![106]
    );
    assert_eq!(
        load(&top(&dense(), "eos_token_id", json!([1, 106]))).eos_token_ids,
        vec![1, 106]
    );
    assert!(load(&dense()).eos_token_ids.is_empty());
}
