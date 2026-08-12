//! Request-policy coverage for the OpenAI / Anthropic surfaces
//! (005 Phase 4.1).
//!
//! `types.rs` is where external JSON becomes a decision, and the branches
//! coverage flagged were the two that matter most for that:
//!
//! * **The imatrix-AWQ thinking override.** A safety net that forces reasoning
//!   off on builds whose calibration corpus lacks reasoning samples, because
//!   leaving it on produces channel-open runaway — the model opens a thinking
//!   channel and never closes it, burning the whole `max_tokens` budget. It is
//!   spelled independently on the OpenAI and Anthropic paths and **neither was
//!   ever exercised**.
//! * **The image size validator.** Every reject arm was uncovered, so a
//!   non-multiple-of-16 or negative dimension had nothing stopping it from
//!   reaching the diffusion pipeline.
//!
//! Requests are built from JSON rather than struct literals on purpose: that is
//! the path a real request takes, and it exercises the serde defaults at the
//! same time.

use lumen_server::types::{AnthropicRequest, ChatCompletionRequest, ImageGenerationRequest};

fn chat(v: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(v).expect("fixture must deserialize")
}

fn anthropic(v: serde_json::Value) -> AnthropicRequest {
    serde_json::from_value(v).expect("fixture must deserialize")
}

fn image(v: serde_json::Value) -> ImageGenerationRequest {
    serde_json::from_value(v).expect("fixture must deserialize")
}

fn openai_base(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": "hi" }],
    })
}

fn anthropic_base(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "hi" }],
    })
}

// ─────────────── imatrix-AWQ override, both API surfaces ───────────────

/// The override outranks every opt-in, on both paths. This is the whole point
/// of it being a safety net rather than a default: a client that asks for
/// thinking on one of these builds gets a runaway, so the request loses.
#[test]
fn the_imatrix_awq_override_beats_every_client_opt_in() {
    let awq_models = [
        "hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq",
        "some/model-AWQ",
        "IMATRIX-flavoured-build",
        // A local-path spelling of the id — the matcher has to see through it
        // too. Written without a home directory on purpose: a `/Users/...`
        // literal in a committed file reads as someone's machine, which is the
        // thing `cargo xtask gate`'s hygiene rule exists to keep out.
        "/models/local-imatrix3plus-awq-4bit",
    ];

    for m in awq_models {
        // OpenAI: every route that would otherwise turn thinking on.
        let mut req = openai_base(m);
        req["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": true });
        assert!(
            !chat(req).enable_thinking_with_backend_default(true),
            "{m}: explicit chat_template_kwargs must not defeat the override"
        );

        let mut req = openai_base(m);
        req["reasoning_effort"] = serde_json::json!("high");
        assert!(
            !chat(req).enable_thinking_with_backend_default(true),
            "{m}: reasoning_effort must not defeat the override"
        );

        let mut req = openai_base(m);
        req["thinking"] = serde_json::json!(true);
        assert!(!chat(req).enable_thinking_with_backend_default(true), "{m}");

        assert!(
            !chat(openai_base(m)).enable_thinking_with_backend_default(true),
            "{m}: the backend default must not defeat the override either"
        );

        // Anthropic: same net, separate implementation.
        let mut req = anthropic_base(m);
        req["thinking"] = serde_json::json!({ "type": "enabled" });
        assert!(
            !anthropic(req).enable_thinking(),
            "{m}: the Anthropic path needs its own copy of the override"
        );

        let mut req = anthropic_base(m);
        req["thinking"] = serde_json::json!(true);
        assert!(!anthropic(req).enable_thinking(), "{m}");
    }
}

/// The detector matches on `imatrix` **or** `-awq`; both operands must fire on
/// their own, and ordinary community quants must not match. A detector that
/// over-matched would silently disable reasoning on models that support it.
#[test]
fn the_detector_matches_both_markers_and_nothing_else() {
    // `-awq` alone, `imatrix` alone.
    for m in ["vendor/model-awq", "vendor/model-imatrix-4bit"] {
        let mut req = openai_base(m);
        req["thinking"] = serde_json::json!(true);
        assert!(
            !chat(req).enable_thinking_with_backend_default(false),
            "{m} should be detected"
        );
    }

    // Community uniform quants and plain ids must NOT be caught.
    for m in [
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        "Qwen/Qwen3.6-27B",
        "vendor/awqmodel",
        "vendor/model-4bit",
        "",
    ] {
        let mut req = openai_base(m);
        req["thinking"] = serde_json::json!(true);
        assert!(
            chat(req).enable_thinking_with_backend_default(false),
            "{m:?} must not be mistaken for an imatrix-AWQ build — the override \
             would silently disable reasoning on a model that supports it"
        );
    }
}

// ─────────────── thinking precedence on a normal model ───────────────

/// The documented precedence, checked rung by rung on a model the override does
/// not catch. Each level has to be able to win over the one below it, or a
/// client's explicit request is quietly ignored.
#[test]
fn thinking_precedence_runs_top_down() {
    const M: &str = "vendor/normal-model";

    // 2. chat_template_kwargs is explicit and beats everything below.
    for want in [true, false] {
        let mut req = openai_base(M);
        req["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": want });
        req["reasoning_effort"] = serde_json::json!("high");
        req["thinking"] = serde_json::json!(true);
        assert_eq!(
            chat(req).enable_thinking_with_backend_default(!want),
            want,
            "explicit chat_template_kwargs must win"
        );
    }

    // A kwargs block with no `enable_thinking` must fall through rather than
    // being read as "false".
    let mut req = openai_base(M);
    req["chat_template_kwargs"] = serde_json::json!({});
    req["thinking"] = serde_json::json!(true);
    assert!(
        chat(req).enable_thinking_with_backend_default(false),
        "an empty kwargs block must not swallow the lower rungs"
    );

    // 3. reasoning_effort: on for real efforts, off for the disabling set,
    //    and it beats the flat flag below it.
    for eff in ["high", "medium", "low", "  HIGH  "] {
        let mut req = openai_base(M);
        req["reasoning_effort"] = serde_json::json!(eff);
        assert!(
            chat(req).enable_thinking_with_backend_default(false),
            "reasoning_effort={eff:?} should enable"
        );
    }
    for eff in ["minimal", "none", "off", "disabled", "", "  None  "] {
        let mut req = openai_base(M);
        req["reasoning_effort"] = serde_json::json!(eff);
        req["thinking"] = serde_json::json!(true);
        assert!(
            !chat(req).enable_thinking_with_backend_default(true),
            "reasoning_effort={eff:?} should disable and outrank thinking:true"
        );
    }

    // 4/5/6: the flat flag, then the operator default, then off.
    let mut req = openai_base(M);
    req["thinking"] = serde_json::json!(true);
    assert!(chat(req).enable_thinking_with_backend_default(false));

    assert!(chat(openai_base(M)).enable_thinking_with_backend_default(true));
    assert!(!chat(openai_base(M)).enable_thinking_with_backend_default(false));
}

/// Anthropic's `thinking` is untagged: an object with `type` or a bare bool.
/// Both spellings, plus the case-insensitive `type` compare.
#[test]
fn anthropic_thinking_accepts_both_spellings() {
    const M: &str = "vendor/normal-model";
    for (v, want) in [
        (serde_json::json!({ "type": "enabled" }), true),
        (serde_json::json!({ "type": "ENABLED" }), true),
        (serde_json::json!({ "type": "disabled" }), false),
        (serde_json::json!({ "type": "anything-else" }), false),
        (serde_json::json!(true), true),
        (serde_json::json!(false), false),
    ] {
        let mut req = anthropic_base(M);
        req["thinking"] = v.clone();
        assert_eq!(anthropic(req).enable_thinking(), want, "thinking={v}");
    }
}

// ─────────────────────── image size validation ───────────────────────

/// Absent size is the documented 1024×1024 default, and valid sizes pass
/// through. Establishing this first keeps the reject table below from passing
/// against a validator that rejects everything.
#[test]
fn valid_sizes_parse_and_absent_means_the_default() {
    assert_eq!(
        image(serde_json::json!({ "prompt": "p" }))
            .dimensions()
            .expect("absent size defaults"),
        (1024, 1024)
    );
    for (s, want) in [
        ("512x512", (512, 512)),
        ("1024x768", (1024, 768)),
        ("16x16", (16, 16)),
        (" 512 x 512 ", (512, 512)),
    ] {
        assert_eq!(
            image(serde_json::json!({ "prompt": "p", "size": s }))
                .dimensions()
                .unwrap_or_else(|e| panic!("{s:?} should parse: {e}")),
            want
        );
    }
}

/// Every reject arm. These had no coverage at all, so nothing stopped a
/// negative or non-multiple-of-16 dimension from reaching the pipeline, where
/// the failure is a kernel-level shape error far from its cause.
#[test]
fn every_invalid_size_is_rejected_with_a_reason() {
    let cases = [
        // Not a multiple of 16 — width, height, and both.
        ("513x512", "multiples of 16"),
        ("512x513", "multiples of 16"),
        ("100x100", "multiples of 16"),
        // Zero and negative.
        ("0x512", "multiples of 16"),
        ("512x0", "multiples of 16"),
        ("-512x512", "multiples of 16"),
        ("512x-512", "multiples of 16"),
        // Malformed: no separator, non-numeric halves.
        ("512", "expected WxH"),
        ("", "expected WxH"),
        ("axb", "invalid width"),
        ("512xb", "invalid height"),
        ("x512", "invalid width"),
        ("512x", "invalid height"),
    ];
    for (s, needle) in cases {
        let err = image(serde_json::json!({ "prompt": "p", "size": s }))
            .dimensions()
            .expect_err(&format!("{s:?} must be rejected"));
        assert!(
            err.contains(needle),
            "{s:?} should be rejected with {needle:?}, got {err:?}"
        );
        assert!(
            err.contains(s) || s.is_empty(),
            "the error should quote the offending value: {err:?}"
        );
    }
}

// ─────────────── parallel_tool_calls, both API surfaces ───────────────
//
// `tool_choice` decides *whether* a tool is called; `parallel_tool_calls`
// decides *how many*. Before these landed the second question had no answer on
// either surface: the field was not declared, `ChatCompletionRequest` has no
// `deny_unknown_fields`, and serde dropped it. A client asking for one call got
// a 200, however many calls the model produced, and no indication that what it
// asked for had been discarded.

/// Absent means `true` — the documented OpenAI default. This is the assertion
/// that keeps a future `#[serde(default)]` edit from quietly making absence
/// mean `false`.
#[test]
fn an_absent_parallel_tool_calls_is_none_not_false() {
    let req = chat(openai_base("m"));
    assert_eq!(req.parallel_tool_calls, None);
    assert_eq!(req.sampling_overrides().parallel_tool_calls, None);
}

/// Both explicit values survive deserialization and reach the overrides the
/// backend actually reads.
#[test]
fn an_explicit_parallel_tool_calls_reaches_the_backend() {
    for want in [true, false] {
        let mut v = openai_base("m");
        v["parallel_tool_calls"] = serde_json::json!(want);
        let req = chat(v);
        assert_eq!(req.parallel_tool_calls, Some(want));
        assert_eq!(
            req.sampling_overrides().parallel_tool_calls,
            Some(want),
            "parallel_tool_calls={want} must not be dropped between request and backend"
        );
    }
}

/// Anthropic spells it inverted and hangs it off `tool_choice` rather than off
/// the request, on every variant. It has to arrive as the same value.
#[test]
fn anthropic_disable_parallel_tool_use_is_inverted_into_the_same_field() {
    for (tc, want) in [
        (
            serde_json::json!({"type": "any", "disable_parallel_tool_use": true}),
            Some(false),
        ),
        (
            serde_json::json!({"type": "any", "disable_parallel_tool_use": false}),
            Some(true),
        ),
        (
            serde_json::json!({"type": "auto", "disable_parallel_tool_use": true}),
            Some(false),
        ),
        (
            serde_json::json!({"type": "tool", "name": "t", "disable_parallel_tool_use": true}),
            Some(false),
        ),
        // Unspecified stays unspecified rather than defaulting to either side.
        (serde_json::json!({"type": "any"}), None),
    ] {
        let mut v = anthropic_base("m");
        v["tool_choice"] = tc.clone();
        assert_eq!(
            anthropic(v).sampling_overrides().parallel_tool_calls,
            want,
            "tool_choice={tc}"
        );
    }
}

/// A request that never mentions tool_choice has nothing to invert.
#[test]
fn anthropic_without_a_tool_choice_leaves_the_count_unspecified() {
    let req = anthropic(anthropic_base("m"));
    assert_eq!(req.sampling_overrides().parallel_tool_calls, None);
}
