//! Phase 1.6.b: robustness regression tests for the model-output-side
//! tool-call body parser (`parse_tool_call_body`). Pinned wire shapes
//! the model has been observed to emit (or could emit under noisy
//! sampling) — we want graceful behavior (best-effort parse OR
//! deterministic error) instead of panic / out-of-bounds.
//!
//! Pairs with `gemma4_tool_call_body.rs` which covers the assistant-
//! side serializer; this file pins the inverse direction.

#![cfg(feature = "mlx-native")]

use lumen_mlx::gemma4::{gemma4_args_to_json, parse_tool_call_body};
use serde_json::json;

#[test]
fn well_formed_single_call_parses_cleanly() {
    let calls =
        parse_tool_call_body("call:get_weather{location:<|\"|>Seoul<|\"|>}").expect("parses");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, json!({"location": "Seoul"}));
}

#[test]
fn parallel_calls_in_one_block_parse() {
    // mlx_lm allows multiple `call:NAME{...}` blocks in one
    // `<|tool_call>...<tool_call|>` span. Parser handles up to N
    // (~8 verified in practice).
    let body = "call:get_weather{location:<|\"|>Seoul<|\"|>}call:get_weather{location:<|\"|>Tokyo<|\"|>}";
    let calls = parse_tool_call_body(body).expect("parses");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].arguments, json!({"location": "Seoul"}));
    assert_eq!(calls[1].arguments, json!({"location": "Tokyo"}));
}

#[test]
fn whitespace_between_name_and_brace_tolerated() {
    let calls =
        parse_tool_call_body("call:noop  {x:1}").expect("whitespace between name and brace ok");
    assert_eq!(calls[0].name, "noop");
    assert_eq!(calls[0].arguments, json!({"x": 1}));
}

#[test]
fn empty_body_errors_not_panic() {
    // No `call:` substring at all — must surface as Err, not panic.
    let err = parse_tool_call_body("").expect_err("empty body should error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no call:NAME") || msg.contains("no call"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn unbalanced_braces_errors_not_panic() {
    // Missing closing brace. Parser should NOT walk off the end of
    // the buffer — anyhow Err is the contract.
    let err = parse_tool_call_body("call:noop{key:42")
        .expect_err("unbalanced brace should error");
    let msg = format!("{err:#}");
    assert!(msg.contains("unbalanced"), "unexpected error: {msg}");
}

#[test]
fn name_only_no_args_block_skipped_gracefully() {
    // `call:noop` with no `{...}` shouldn't crash. Parser advances to
    // the next call or returns Err if no valid calls were found.
    let err = parse_tool_call_body("call:noop").expect_err("name-only should yield no calls");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no call:NAME") || msg.contains("no call"),
        "unexpected error: {msg}"
    );
}

#[test]
fn malformed_first_call_recovers_to_second() {
    // First "call:" is missing the brace; parser should skip and
    // grab the second well-formed call.
    let body = "call:bad_no_brace call:good{x:1}";
    let calls = parse_tool_call_body(body).expect("recovers to second call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "good");
}

#[test]
fn numeric_boolean_array_argument_types() {
    // Gemma 4 emits numbers / booleans / arrays as raw JSON inside
    // the args block (only strings get the `<|"|>` wrapper). The
    // converter must preserve types end-to-end.
    let calls = parse_tool_call_body("call:fn{n:42,f:3.14,b:true,arr:[1,2,3]}").expect("parses");
    assert_eq!(
        calls[0].arguments,
        json!({"n": 42, "f": 3.14, "b": true, "arr": [1, 2, 3]})
    );
}

#[test]
fn nested_object_value_parses() {
    // Gemma 4 can emit nested objects under a key (rare in practice
    // but legal). Verify balanced-brace tracking handles it.
    let body = "call:fn{outer:{inner_str:<|\"|>v<|\"|>,inner_num:7}}";
    let calls = parse_tool_call_body(body).expect("nested parses");
    assert_eq!(
        calls[0].arguments,
        json!({"outer": {"inner_str": "v", "inner_num": 7}})
    );
}

#[test]
fn empty_args_object_parses() {
    let calls = parse_tool_call_body("call:ping{}").expect("parses");
    assert_eq!(calls[0].name, "ping");
    assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn args_to_json_passes_through_strict_json() {
    // gemma4_args_to_json is a thin wrapper that accepts both Gemma
    // pseudo-JSON and strict JSON. Strict input should pass through
    // unchanged. (This guards against accidental double-escaping.)
    let v = gemma4_args_to_json(r#"{"k":"v","n":1}"#).expect("strict JSON parses");
    assert_eq!(v, json!({"k": "v", "n": 1}));
}

#[test]
fn args_to_json_handles_quoted_keys_already() {
    // Some tokenizers emit quoted keys even though Gemma's grammar
    // doesn't require them. Converter should accept both.
    let v = gemma4_args_to_json(r#"{"location":<|"|>Seoul<|"|>}"#)
        .expect("quoted keys + Gemma string delim");
    assert_eq!(v, json!({"location": "Seoul"}));
}
