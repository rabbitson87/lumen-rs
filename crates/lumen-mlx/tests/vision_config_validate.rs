//! The two `vision_config` validators (005 Phase 4.1).
//!
//! These were the largest block of **unreached** code in the named scope — 11
//! branches in the Qwen parser and ~20 in the Gemma 4 one, none of which any
//! tier-0 binary touched. It looked at first like the case for "leave it to
//! red-green": the vision towers need a GPU and weights.
//!
//! They do not. Both validators are pure `serde` + arithmetic over a parsed
//! struct, and they moved into the ungated config modules with the rest of
//! their `config.json`. Nothing about them needs Metal. They were unreached
//! only because nobody had called them.
//!
//! Worth covering rather than excluding, because every rule here is a **shape**
//! contract: `hidden_size % num_heads`, `head_dim % 4` for 2-D RoPE,
//! `√num_position_embeddings` being a whole number. A checkpoint that violates
//! one of those does not fail at load — it fails much later inside a kernel,
//! as a dimension mismatch with nothing pointing back at the config field that
//! caused it.

use lumen_mlx::gemma4_config::NativeGemma4VisionConfig;
use lumen_mlx::qwen35_config::NativeQwen36VisionConfig;
use serde_json::{Value, json};

// ───────────────────────────── Qwen 3.6 ─────────────────────────────

/// A vision block whose numbers are mutually consistent: `hidden_size` divides
/// by `num_heads`, the resulting `head_dim` is a multiple of 4, and
/// `num_position_embeddings` is a perfect square.
fn qwen_vision() -> Value {
    json!({
        "depth": 24,
        "hidden_size": 1024,
        "num_heads": 16,          // head_dim = 64, a multiple of 4
        "intermediate_size": 4096,
        "in_channels": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,  // 48²
        "out_hidden_size": 2048,
    })
}

fn qwen(v: &Value) -> NativeQwen36VisionConfig {
    serde_json::from_value(v.clone()).expect("fixture must deserialize")
}

fn with(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v.as_object_mut().unwrap().insert(key.into(), val);
    v
}

#[test]
fn the_qwen_vision_baseline_validates() {
    qwen(&qwen_vision())
        .validate()
        .expect("the fixture must validate — otherwise every case below is noise");
}

#[test]
fn qwen_vision_rejects_every_zero_core_dim_individually() {
    for field in [
        "depth",
        "hidden_size",
        "num_heads",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
    ] {
        let err = qwen(&with(&qwen_vision(), field, json!(0)))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero-valued"), "{field}=0 gave: {err}");
    }
}

/// The shape rules. Each one is a division or a square root that a later
/// kernel performs unchecked, so violating it here is the difference between a
/// named config error and an opaque dimension mismatch mid-forward.
#[test]
fn qwen_vision_enforces_its_shape_arithmetic() {
    // hidden_size must divide by num_heads.
    let err = qwen(&with(&qwen_vision(), "num_heads", json!(15)))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("divisible"), "got: {err}");

    // head_dim must be a multiple of 4: the rotary table is built over
    // head_dim/2 and split again for the (h, w) pair.
    // 1024 / 32 = 32 (ok); 1024 / 512 = 2 (not a multiple of 4).
    let err = qwen(&with(&qwen_vision(), "num_heads", json!(512)))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("multiple of 4"), "got: {err}");

    // num_position_embeddings must be a perfect square — it is a 2-D grid.
    let err = qwen(&with(
        &qwen_vision(),
        "num_position_embeddings",
        json!(2305),
    ))
    .validate()
    .unwrap_err()
    .to_string();
    assert!(err.contains("perfect square"), "got: {err}");

    // ...and the accessors that depend on those rules agree with them.
    let c = qwen(&qwen_vision());
    assert_eq!(c.head_dim(), 64);
    assert_eq!(c.merge_unit(), 4, "merge² folds 4 patches into one token");
    assert_eq!(c.grid_per_side(), 48);
}

/// Two unsupported-feature rejections. Both are "we have not implemented this
/// yet" rather than "your config is wrong", and both must say so rather than
/// loading and producing wrong features.
#[test]
fn qwen_vision_rejects_the_features_it_does_not_implement() {
    let err = qwen(&with(
        &qwen_vision(),
        "deepstack_visual_indexes",
        json!([1, 2]),
    ))
    .validate()
    .unwrap_err()
    .to_string();
    assert!(err.contains("deepstack"), "got: {err}");

    // An empty list is the supported case and must pass.
    qwen(&with(&qwen_vision(), "deepstack_visual_indexes", json!([])))
        .validate()
        .expect("an empty deepstack list is the shipped shape");

    let err = qwen(&with(&qwen_vision(), "hidden_act", json!("silu")))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("hidden_act"), "got: {err}");

    // The default activation is the supported one.
    qwen(&with(
        &qwen_vision(),
        "hidden_act",
        json!("gelu_pytorch_tanh"),
    ))
    .validate()
    .expect("the default activation must validate");
}

// ───────────────────────────── Gemma 4 ─────────────────────────────

fn gemma_vision() -> Value {
    json!({
        "model_type": "gemma4_vision",
        "hidden_size": 1152,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "head_dim": 72,          // 16 × 72 = 1152
        "intermediate_size": 4304,
        "patch_size": 14,
        "pooling_kernel_size": 2,
        "position_embedding_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": { "rope_theta": 10000.0 },
    })
}

fn gemma(v: &Value) -> NativeGemma4VisionConfig {
    serde_json::from_value(v.clone()).expect("fixture must deserialize")
}

#[test]
fn the_gemma_vision_baseline_validates() {
    gemma(&gemma_vision())
        .validate()
        .expect("the fixture must validate — otherwise every case below is noise");
}

#[test]
fn gemma_vision_checks_its_identity_and_dims() {
    for mt in ["gemma3_vision", "siglip", ""] {
        let err = gemma(&with(&gemma_vision(), "model_type", json!(mt)))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_type"), "{mt:?} gave: {err}");
    }

    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "patch_size",
        "pooling_kernel_size",
    ] {
        assert!(
            gemma(&with(&gemma_vision(), field, json!(0)))
                .validate()
                .is_err(),
            "{field}=0 must be rejected"
        );
    }
}

/// `num_attention_heads × head_dim` must equal `hidden_size`. Getting this
/// wrong reshapes attention into the wrong number of heads — the tower still
/// runs, and the features it produces are nonsense.
#[test]
fn gemma_vision_enforces_the_head_product() {
    let err = gemma(&with(&gemma_vision(), "head_dim", json!(64)))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("hidden_size"), "got: {err}");

    // A consistent alternative shape must still pass, so the rule is an
    // equality rather than a hard-coded pair.
    let v = with(
        &with(&gemma_vision(), "head_dim", json!(64)),
        "hidden_size",
        json!(1024),
    );
    gemma(&v).validate().expect("16 × 64 == 1024 is consistent");
}

#[test]
fn gemma_vision_rejects_the_features_it_does_not_implement() {
    let err = gemma(&with(&gemma_vision(), "use_clipped_linears", json!(true)))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("use_clipped_linears"), "got: {err}");

    let err = gemma(&with(&gemma_vision(), "hidden_activation", json!("relu")))
        .validate()
        .unwrap_err()
        .to_string();
    assert!(err.contains("hidden_activation"), "got: {err}");
}
