//! Contract coverage for the Qwen 3.5/3.6 config validators (005 Phase 4.1).
//!
//! `config_faults.rs` sweeps *corrupt* configs through `load()`. That leaves
//! the validators almost untouched, because `load()` only parses — the checks
//! live in `validate_*`, which is called by the mlx-native model loader. The
//! branch baseline made the gap visible and put a number on it: 98 branches in
//! this module, **98 missed, 0.00%**.
//!
//! The gap matters more than the number. These validators are the last thing
//! between a plausible-looking `config.json` and a model built with the wrong
//! dimensions — a `rope_dim` that is odd, a `layer_types` list one entry short,
//! a `num_experts_per_tok` larger than `num_experts`. None of those crash at
//! load; they produce a model that runs and is wrong, which is the failure
//! class 005 exists for. Every one of them is a pure arithmetic check over a
//! parsed struct, so every one is tier-0 testable.
//!
//! Written table-driven for the same reason the fault sweeps are: the point is
//! case density per line of test code, not a hand-picked example per rule.

use lumen_mlx::qwen35_config::{MlpKind, NativeModelConfig};
use serde_json::{Value, json};

// ───────────────────────────── fixtures ─────────────────────────────

/// A minimal **dense** config that validates. Every mutation below starts here,
/// so if this stops validating the whole file reports noise instead of rules —
/// which is what `the_dense_baseline_validates` guards.
fn dense() -> Value {
    json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
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

/// The MoE shape: `num_experts > 0` selects the other arm of every MLP check.
fn moe() -> Value {
    let mut v = dense();
    let tc = v["text_config"].as_object_mut().unwrap();
    tc.insert("model_type".into(), json!("qwen3_5_moe_text"));
    tc.insert("num_experts".into(), json!(128));
    tc.insert("num_experts_per_tok".into(), json!(8));
    tc.insert("moe_intermediate_size".into(), json!(768));
    v["model_type"] = json!("qwen3_5_moe");
    v
}

fn load(v: &Value) -> NativeModelConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_string(v).unwrap()).expect("write");
    NativeModelConfig::load(&path).expect("fixture must parse")
}

/// Set `text_config.<key>`.
fn tc(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .insert(key.into(), val);
    v
}

/// Set a top-level key.
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
fn the_dense_baseline_validates() {
    let cfg = load(&dense());
    assert_eq!(
        cfg.validate_qwen3_5_family().expect("dense must validate"),
        MlpKind::Dense
    );
    assert_eq!(cfg.text_config.mlp_kind(), MlpKind::Dense);
}

#[test]
fn the_moe_baseline_validates_through_both_entry_points() {
    let cfg = load(&moe());
    assert_eq!(
        cfg.validate_qwen3_5_family().expect("moe family"),
        MlpKind::Moe
    );
    // The strict MoE-only contract is a separate entry point with its own
    // error surface; both must accept the same config.
    cfg.validate_qwen3_5_moe().expect("moe strict");
    assert_eq!(cfg.text_config.mlp_kind(), MlpKind::Moe);
}

// ───────────────────────── family / model_type ─────────────────────────

#[test]
fn model_type_is_checked_at_both_levels_and_by_both_entry_points() {
    // Top-level, family validator: accepts the two known values, rejects others.
    for mt in ["qwen3_5", "qwen3_5_moe"] {
        load(&top(&moe(), "model_type", json!(mt)))
            .validate_qwen3_5_family()
            .unwrap_or_else(|e| panic!("model_type={mt} should be accepted: {e:#}"));
    }
    for mt in ["qwen3", "gemma4", "", "QWEN3_5"] {
        let err = load(&top(&moe(), "model_type", json!(mt)))
            .validate_qwen3_5_family()
            .expect_err("unknown model_type must be rejected");
        assert!(
            format!("{err:#}").contains("model_type"),
            "error should name the field: {err:#}"
        );
    }

    // The strict MoE entry point is narrower: dense's `qwen3_5` is not enough.
    let dense_named = load(&top(&moe(), "model_type", json!("qwen3_5")));
    assert!(
        dense_named.validate_qwen3_5_moe().is_err(),
        "the MoE-only contract must reject the dense model_type even though the \
         family validator accepts it — that difference is the whole reason two \
         entry points exist"
    );

    // text_config.model_type, family validator.
    for mt in ["qwen3_5_text", "qwen3_5_moe_text"] {
        load(&tc(&moe(), "model_type", json!(mt)))
            .validate_qwen3_5_family()
            .unwrap_or_else(|e| panic!("text model_type={mt} should be accepted: {e:#}"));
    }
    let err = load(&tc(&moe(), "model_type", json!("qwen3_text")))
        .validate_qwen3_5_family()
        .expect_err("unknown text model_type must be rejected");
    assert!(format!("{err:#}").contains("text_config.model_type"));

    // ...and the strict validator wants exactly `qwen3_5_moe_text`.
    let err = load(&tc(&moe(), "model_type", json!("qwen3_5_text")))
        .validate_qwen3_5_moe()
        .expect_err("strict MoE requires the MoE text model_type");
    assert!(format!("{err:#}").contains("qwen3_5_moe_text"));
}

// ───────────────────────────── core dims ─────────────────────────────

/// Each of the six dims must be rejected at zero **individually**. Testing one
/// of them would pass with a validator that checks only that one, which is the
/// bug a short-circuiting `||` chain invites.
#[test]
fn every_core_dim_is_rejected_at_zero_individually() {
    for field in [
        "hidden_size",
        "head_dim",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
    ] {
        let mut v = tc(&dense(), field, json!(0));
        // `num_hidden_layers = 0` would also trip the layer_types check, so
        // keep that consistent and let the zero-dim rule be the one that fires.
        if field == "num_hidden_layers" {
            v = tc(&v, "layer_types", json!([]));
        }
        assert!(
            load(&v).validate_qwen3_5_family().is_err(),
            "{field}=0 must be rejected — a zero dimension does not crash, it \
             builds a mis-shaped model"
        );
    }
}

#[test]
fn zero_dims_report_a_zero_sized_error() {
    for field in [
        "hidden_size",
        "head_dim",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
    ] {
        let err = load(&tc(&dense(), field, json!(0)))
            .validate_qwen3_5_family()
            .expect_err("zero dim must be rejected");
        assert!(
            format!("{err:#}").contains("zero-sized"),
            "{field}=0 gave: {err:#}"
        );
    }
}

#[test]
fn layer_types_length_must_match_the_layer_count() {
    // Short, long, and empty — the boundary is an equality, so both sides count.
    for types in [
        json!(["linear_attention", "linear_attention", "full_attention"]),
        json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "full_attention"
        ]),
        json!([]),
    ] {
        let err = load(&tc(&dense(), "layer_types", types.clone()))
            .validate_qwen3_5_family()
            .expect_err("mismatched layer_types must be rejected");
        assert!(
            format!("{err:#}").contains("layer_types length"),
            "got: {err:#}"
        );
    }
}

/// RoPE rotates coordinate *pairs*, so an odd or zero rotary span is not a
/// slightly-wrong model — it is an out-of-bounds read waiting to happen.
#[test]
fn rope_dim_must_be_a_positive_even_number() {
    // head_dim × partial_rotary_factor: 256 × 0.25 = 64, fine.
    assert_eq!(load(&dense()).text_config.rope_dim(), 64);

    // 0 span (factor rounds to zero).
    let err = load(&tc(&dense(), "partial_rotary_factor", json!(0.001)))
        .validate_qwen3_5_family()
        .expect_err("zero rope_dim must be rejected");
    assert!(format!("{err:#}").contains("rope_dim"), "got: {err:#}");

    // Odd span: head_dim 10 × 0.5 = 5.
    let v = tc(
        &tc(&dense(), "head_dim", json!(10)),
        "partial_rotary_factor",
        json!(0.5),
    );
    let err = load(&v)
        .validate_qwen3_5_family()
        .expect_err("odd rope_dim must be rejected");
    assert!(format!("{err:#}").contains("even"), "got: {err:#}");
}

// ───────────────────────────── MLP arms ─────────────────────────────

#[test]
fn the_moe_arm_checks_its_own_size_fields() {
    // top_k = 0
    let err = load(&tc(&moe(), "num_experts_per_tok", json!(0)))
        .validate_qwen3_5_family()
        .expect_err("top_k=0 must be rejected");
    assert!(format!("{err:#}").contains("num_experts_per_tok"));

    // top_k > num_experts — the off-by-one side of the same rule.
    let err = load(&tc(&moe(), "num_experts_per_tok", json!(129)))
        .validate_qwen3_5_family()
        .expect_err("top_k > num_experts must be rejected");
    assert!(format!("{err:#}").contains("num_experts_per_tok"));

    // top_k == num_experts is legal (dense routing over all experts).
    load(&tc(&moe(), "num_experts_per_tok", json!(128)))
        .validate_qwen3_5_family()
        .expect("top_k == num_experts is a valid boundary, not an error");

    // moe_intermediate_size = 0
    let err = load(&tc(&moe(), "moe_intermediate_size", json!(0)))
        .validate_qwen3_5_family()
        .expect_err("zero moe_intermediate_size must be rejected");
    assert!(format!("{err:#}").contains("moe_intermediate_size"));
}

#[test]
fn the_dense_arm_checks_intermediate_size() {
    let err = load(&tc(&dense(), "intermediate_size", json!(0)))
        .validate_qwen3_5_family()
        .expect_err("dense with zero intermediate_size must be rejected");
    assert!(
        format!("{err:#}").contains("intermediate_size"),
        "got: {err:#}"
    );
}

/// The strict MoE validator has its own copy of the MoE size checks, reached
/// through `num_experts > 0` rather than through `mlp_kind()`. Same rules, a
/// different code path — so covering one says nothing about the other.
#[test]
fn the_strict_moe_validator_repeats_the_size_checks() {
    for (field, val) in [
        ("num_experts_per_tok", json!(0)),
        ("num_experts_per_tok", json!(999)),
        ("moe_intermediate_size", json!(0)),
    ] {
        let err = load(&tc(&moe(), field, val.clone()))
            .validate_qwen3_5_moe()
            .expect_err("strict validator must reject too");
        assert!(format!("{err:#}").contains(field), "{field}={val}: {err:#}");
    }
    // With num_experts == 0 the strict validator skips the MoE block entirely.
    load(&tc(
        &tc(&moe(), "num_experts", json!(0)),
        "num_experts_per_tok",
        json!(0),
    ))
    .validate_qwen3_5_moe()
    .expect("num_experts=0 skips the MoE size checks");
}

// ───────────────────────────── quantization ─────────────────────────────

#[test]
fn the_family_validator_accepts_the_shipped_quant_recipes() {
    for (mode, bits) in [
        ("mxfp4", 4),
        ("nvfp4", 4),
        ("affine", 4),
        ("affine", 6),
        ("affine", 8),
    ] {
        load(&top(&moe(), "quantization_config", quant(mode, bits, 64)))
            .validate_qwen3_5_family()
            .unwrap_or_else(|e| panic!("{mode}/{bits}-bit ships in production: {e:#}"));
    }
}

#[test]
fn the_family_validator_rejects_unshippable_quant_recipes() {
    let bad = [
        // unknown mode
        ("int4", 4, 64),
        ("", 4, 64),
        // E2M1 formats are 4-bit only
        ("mxfp4", 8, 64),
        ("nvfp4", 6, 64),
        // affine's allowed set excludes these
        ("affine", 2, 64),
        ("affine", 3, 64),
        ("affine", 5, 64),
        ("affine", 16, 64),
        // group_size 0 is a division by zero downstream
        ("affine", 4, 0),
        ("mxfp4", 4, 0),
    ];
    for (mode, bits, group) in bad {
        let err = load(&top(
            &moe(),
            "quantization_config",
            quant(mode, bits, group),
        ))
        .validate_qwen3_5_family()
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("quantization_config"),
            "{mode}/{bits}/g{group} gave: {err:#}"
        );
    }
}

/// The strict MoE validator is deliberately narrower than the family one:
/// affine is a dense-family recipe and must not pass here.
#[test]
fn the_strict_validator_is_narrower_than_the_family_one() {
    let affine = top(&moe(), "quantization_config", quant("affine", 4, 64));
    load(&affine)
        .validate_qwen3_5_family()
        .expect("family accepts affine");
    assert!(
        load(&affine).validate_qwen3_5_moe().is_err(),
        "the strict MoE contract must reject affine — if both validators agreed \
         there would be no reason for the second one to exist"
    );

    for (mode, bits, group) in [("mxfp4", 4, 64), ("nvfp4", 4, 32)] {
        load(&top(
            &moe(),
            "quantization_config",
            quant(mode, bits, group),
        ))
        .validate_qwen3_5_moe()
        .unwrap_or_else(|e| panic!("{mode} must pass the strict validator: {e:#}"));
    }
    for (mode, bits, group) in [("mxfp4", 8, 64), ("mxfp4", 4, 0)] {
        assert!(
            load(&top(
                &moe(),
                "quantization_config",
                quant(mode, bits, group)
            ))
            .validate_qwen3_5_moe()
            .is_err(),
            "{mode}/{bits}/g{group} must be rejected by the strict validator"
        );
    }
}

#[test]
fn an_absent_quantization_block_is_fine() {
    // bf16 checkpoints carry no quantization_config at all.
    load(&dense())
        .validate_qwen3_5_family()
        .expect("unquantized config must validate");
    load(&moe())
        .validate_qwen3_5_moe()
        .expect("unquantized MoE config must validate");
}

// ───────────────────────────── accessors ─────────────────────────────

#[test]
fn mrope_returns_none_unless_the_sections_are_usable() {
    let base = dense(); // rope_dim 64 → sections must sum to 32

    // No rope_parameters at all.
    assert!(load(&base).text_config.mrope().is_none());

    // rope_parameters present, no mrope_section.
    let v = tc(&base, "rope_parameters", json!({}));
    assert!(load(&v).text_config.mrope().is_none());

    // Wrong length.
    for sections in [json!([32]), json!([11, 11]), json!([8, 8, 8, 8])] {
        let v = tc(
            &base,
            "rope_parameters",
            json!({ "mrope_section": sections }),
        );
        assert!(
            load(&v).text_config.mrope().is_none(),
            "a section list that is not 3 long must be None, not an error"
        );
    }

    // Right length, wrong sum.
    let v = tc(
        &base,
        "rope_parameters",
        json!({ "mrope_section": [11, 11, 11] }),
    );
    assert!(load(&v).text_config.mrope().is_none());

    // Usable: 11 + 11 + 10 == 32 == rope_dim / 2.
    let v = tc(
        &base,
        "rope_parameters",
        json!({ "mrope_section": [11, 11, 10], "mrope_interleaved": true }),
    );
    let (sections, interleaved) = load(&v).text_config.mrope().expect("usable mrope");
    assert_eq!(sections, [11, 11, 10]);
    assert!(interleaved);

    // ...and a text-only deploy must still LOAD with an unusable section list:
    // refusing would break text serving over a field only images use.
    let v = tc(
        &base,
        "rope_parameters",
        json!({ "mrope_section": [1, 2, 3] }),
    );
    load(&v)
        .validate_qwen3_5_family()
        .expect("an unusable mrope_section must not fail validation");
}

#[test]
fn layer_kind_accessors_read_the_layer_types_list() {
    let cfg = load(&dense());
    assert_eq!(
        cfg.text_config.is_linear_per_layer(),
        vec![true, true, true, false]
    );
    assert_eq!(cfg.text_config.first_full_attn_layer(), Some(3));

    // All-linear: no full-attention layer to find.
    let v = tc(
        &dense(),
        "layer_types",
        json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention"
        ]),
    );
    let cfg = load(&v);
    assert_eq!(cfg.text_config.first_full_attn_layer(), None);
    assert!(cfg.text_config.is_linear_per_layer().iter().all(|&b| b));

    // All-full: index 0.
    let v = tc(
        &dense(),
        "layer_types",
        json!([
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ]),
    );
    let cfg = load(&v);
    assert_eq!(cfg.text_config.first_full_attn_layer(), Some(0));
    assert!(cfg.text_config.is_linear_per_layer().iter().all(|&b| !b));
}

/// `eos_token_id` is spelled as a scalar by some exporters and a list by
/// others; both must land on the same shape.
#[test]
fn eos_token_id_accepts_both_a_scalar_and_a_list() {
    let one = load(&top(&dense(), "eos_token_id", json!(151645)));
    assert_eq!(one.eos_token_ids, vec![151645]);

    let many = load(&top(&dense(), "eos_token_id", json!([151645, 151643])));
    assert_eq!(many.eos_token_ids, vec![151645, 151643]);

    // Absent → empty, not a parse failure.
    assert!(load(&dense()).eos_token_ids.is_empty());
}

/// EOS is spelled at the **top level** by some checkpoints and inside
/// `text_config` by others, and reading only one of them does not error — it
/// yields an EMPTY stop set, and generation runs past the turn boundary into
/// the next turn's header. `Qwen3.5-9B-MTPLX-Speed` is a shipping checkpoint
/// that declares it only in the nested block.
///
/// Found by the by-hand end-to-end request in `docs/release-checklist.md` §5,
/// which is exactly the failure automation cannot see: the answer is correct,
/// and then it keeps going.
#[test]
fn eos_token_id_is_read_from_either_level() {
    // Top level only — the common spelling.
    let cfg = load(&top(&dense(), "eos_token_id", json!([151645])));
    assert_eq!(cfg.eos_token_ids, vec![151645]);
    assert!(cfg.text_config.eos_token_ids.is_empty());

    // Nested only — the shape that produced an empty stop set.
    let cfg = load(&tc(&dense(), "eos_token_id", json!(248044)));
    assert_eq!(
        cfg.text_config.eos_token_ids,
        vec![248044],
        "a checkpoint declaring EOS only in text_config must still be readable, \
         or nothing terminates the turn"
    );

    // Both, with the top level richer — the Gemma-style layout where the top
    // level carries extra stop tokens the nested block omits.
    let v = tc(
        &top(&dense(), "eos_token_id", json!([1, 106, 50])),
        "eos_token_id",
        json!(1),
    );
    let cfg = load(&v);
    assert_eq!(cfg.eos_token_ids, vec![1, 106, 50]);
    assert_eq!(cfg.text_config.eos_token_ids, vec![1]);

    // Absent from both is empty rather than a parse failure — an unquantized
    // or minimal config must still load.
    let cfg = load(&dense());
    assert!(cfg.eos_token_ids.is_empty() && cfg.text_config.eos_token_ids.is_empty());
}
