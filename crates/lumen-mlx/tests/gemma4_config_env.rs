//! Load-time environment overrides for the Gemma 4 config (005 Phase 4.1).
//!
//! **Its own test binary on purpose.** `LUMEN_SLIDING_WINDOW` and
//! `LUMEN_MAX_CTX` are read by `NativeGemma4Config::load()` from the process
//! environment, so a test that sets them races every other test in the same
//! binary that calls `load()`. Concretely: with the override active,
//! `gemma4_config_validate`'s `sliding_window = 0` case would be rewritten to a
//! valid window and its expected error would vanish. That is a flaky test
//! waiting to happen, and one test binary per process is the only fix that does
//! not depend on `--test-threads=1` being remembered.
//!
//! These two vars deserve coverage more than most config rules do: they are the
//! only place a *valid* config yields a different model depending on the
//! environment, with nothing in the file to explain the difference.

use lumen_mlx::gemma4_config::NativeGemma4Config;
use serde_json::{Value, json};

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

fn tc(v: &Value, key: &str, val: Value) -> Value {
    let mut v = v.clone();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .insert(key.into(), val);
    v
}

fn load(v: &Value) -> NativeGemma4Config {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_string(v).unwrap()).expect("write");
    NativeGemma4Config::load(&path).expect("fixture must parse")
}

/// `LUMEN_SLIDING_WINDOW` and `LUMEN_MAX_CTX` let the desktop app rewrite two
/// model dimensions at load time. They are the only place in this file where a
/// *valid* config produces a different model depending on the environment, so
/// they deserve a test more than most of the rules above do: a silent override
/// that mis-parses, or clamps the wrong way, changes attention behaviour with
/// nothing in the config to explain it.
///
/// Both live in one test on purpose — they mutate process-global state, and
/// splitting them would let two tests in this binary race.
#[test]
fn the_load_time_env_overrides_apply_clamp_and_ignore_correctly() {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: single-threaded within this test; both vars are read by
            // `load()` on the next call rather than cached.
            unsafe {
                std::env::remove_var("LUMEN_SLIDING_WINDOW");
                std::env::remove_var("LUMEN_MAX_CTX");
            }
        }
    }
    let _g = Guard;
    // SAFETY: as above.
    let set = |k: &str, v: &str| unsafe { std::env::set_var(k, v) };
    let clear = |k: &str| unsafe { std::env::remove_var(k) };

    // Baseline: no overrides.
    clear("LUMEN_SLIDING_WINDOW");
    clear("LUMEN_MAX_CTX");
    let cfg = load(&dense());
    assert_eq!(cfg.text_config.sliding_window, 512);
    assert_eq!(cfg.text_config.max_position_embeddings, 4096);

    // Sliding window: applied verbatim, in both directions. Unlike the ctx cap
    // this is not a clamp — the app is choosing the window, not bounding it.
    for want in ["256", "2048"] {
        set("LUMEN_SLIDING_WINDOW", want);
        assert_eq!(
            load(&dense()).text_config.sliding_window,
            want.parse::<usize>().unwrap()
        );
    }

    // Ignored inputs must leave the config's own value alone rather than
    // zeroing it: `0` is the documented "no override", and a non-numeric value
    // is a typo, not an instruction.
    for ignored in ["0", "", "abc", "-1", "512tokens"] {
        set("LUMEN_SLIDING_WINDOW", ignored);
        assert_eq!(
            load(&dense()).text_config.sliding_window,
            512,
            "LUMEN_SLIDING_WINDOW={ignored:?} must be ignored, not applied"
        );
    }
    clear("LUMEN_SLIDING_WINDOW");

    // Max context: a **cap**, so it only ever lowers.
    set("LUMEN_MAX_CTX", "2048");
    assert_eq!(load(&dense()).text_config.max_position_embeddings, 2048);

    set("LUMEN_MAX_CTX", "65536");
    assert_eq!(
        load(&dense()).text_config.max_position_embeddings,
        4096,
        "a cap above the model's own value must not raise it — the weights \
         cannot serve a longer context just because an env var asked"
    );

    // Equal is not a lowering, so it is also a no-op.
    set("LUMEN_MAX_CTX", "4096");
    assert_eq!(load(&dense()).text_config.max_position_embeddings, 4096);

    for ignored in ["0", "", "nope"] {
        set("LUMEN_MAX_CTX", ignored);
        assert_eq!(
            load(&dense()).text_config.max_position_embeddings,
            4096,
            "LUMEN_MAX_CTX={ignored:?} must be ignored"
        );
    }

    // An overridden config must still validate — the override runs inside
    // `load()`, before any caller gets a chance to check it.
    set("LUMEN_SLIDING_WINDOW", "128");
    set("LUMEN_MAX_CTX", "1024");
    let cfg = load(&dense());
    cfg.validate_gemma4_family()
        .expect("an env-overridden config must still be valid");
    assert_eq!(cfg.text_config.sliding_window, 128);
    assert_eq!(cfg.text_config.max_position_embeddings, 1024);
}

/// `LUMEN_GEMMA4_TOP_K` overrides the MoE routing width at load time — the
/// third of this config's load-time env knobs and the one that changes *model
/// behaviour* rather than a memory bound. A router that suddenly picks a
/// different number of experts produces different text with nothing in the
/// config to explain it.
#[test]
fn the_top_k_override_applies_and_ignores_junk() {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: single-threaded within this test binary.
            unsafe { std::env::remove_var("LUMEN_GEMMA4_TOP_K") };
        }
    }
    let _g = Guard;
    // SAFETY: as above.
    let set = |v: &str| unsafe { std::env::set_var("LUMEN_GEMMA4_TOP_K", v) };
    let clear = || unsafe { std::env::remove_var("LUMEN_GEMMA4_TOP_K") };

    let moe = {
        let v = dense();
        let v = tc(&v, "enable_moe_block", json!(true));
        let v = tc(&v, "num_experts", json!(128));
        let v = tc(&v, "top_k_experts", json!(8));
        tc(&v, "moe_intermediate_size", json!(1024))
    };

    clear();
    assert_eq!(
        load(&moe).text_config.top_k_experts,
        8,
        "unset means the config's own value"
    );

    for want in ["1", "4", "16"] {
        set(want);
        assert_eq!(
            load(&moe).text_config.top_k_experts,
            want.parse::<usize>().unwrap(),
            "LUMEN_GEMMA4_TOP_K={want} must be applied"
        );
    }

    for junk in ["0", "", "eight", "-1", "4.5"] {
        set(junk);
        assert_eq!(
            load(&moe).text_config.top_k_experts,
            8,
            "LUMEN_GEMMA4_TOP_K={junk:?} must be ignored, not zero the router"
        );
    }

    // An override equal to the config's own value is a no-op rather than a
    // logged change — the `n != top_k_experts` guard's other side, and the
    // shape an operator gets when they pin the value the model already uses.
    set("8");
    assert_eq!(load(&moe).text_config.top_k_experts, 8);

    // An overridden config must still validate — the override runs inside
    // `load()`, before any caller can check it.
    set("4");
    load(&moe)
        .validate_gemma4_family()
        .expect("an overridden top_k must still be a valid config");
}
