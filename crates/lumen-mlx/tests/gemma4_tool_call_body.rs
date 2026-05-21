//! Pure-text tests for the assistant-side `<|tool_call>call:NAME{...}`
//! body formatter (`format_tool_call_body`). Pins the wire format
//! independently of the tokenizer so the turn-2 stitching code can be
//! validated without loading a real tokenizer.json.
//!
//! The companion `render_chat_history` is integration-tested with a real
//! tokenizer; this file only covers the pure-Rust serialization.

#![cfg(feature = "mlx-native")]

use lumen_mlx::gemma4::format_tool_call_body;
use serde_json::json;

#[test]
fn empty_arguments() {
    let body = format_tool_call_body("ping", &json!({}));
    assert_eq!(body, "call:ping{}");
}

#[test]
fn null_or_non_object_arguments_render_empty_body() {
    // Defensive: model output occasionally drops to a non-object value.
    // Renderer must not panic and emits an empty body.
    assert_eq!(format_tool_call_body("noop", &json!(null)), "call:noop{}");
    assert_eq!(format_tool_call_body("noop", &json!(42)), "call:noop{}");
    assert_eq!(format_tool_call_body("noop", &json!("str")), "call:noop{}");
}

#[test]
fn single_string_argument() {
    let body = format_tool_call_body("get_weather", &json!({"location": "Seoul"}));
    // Keys bare, string values wrapped with <|"|>
    assert_eq!(body, "call:get_weather{location:<|\"|>Seoul<|\"|>}");
}

#[test]
fn mixed_value_types() {
    let body = format_tool_call_body(
        "config",
        &json!({
            "name": "alpha",
            "count": 7,
            "enabled": true,
            "ratio": 0.5
        }),
    );
    // Insertion order is preserved (serde_json::Map preserves with the
    // `preserve_order` feature; lumen workspace uses standard Map which
    // is BTreeMap-backed in default features → keys end up sorted, but
    // we don't assert key order beyond "string value escaped, others not").
    assert!(body.starts_with("call:config{"));
    assert!(body.ends_with('}'));
    assert!(body.contains("name:<|\"|>alpha<|\"|>"));
    assert!(body.contains("count:7"));
    assert!(body.contains("enabled:true"));
    assert!(body.contains("ratio:0.5"));
}

#[test]
fn nested_object_value() {
    // Nested mapping renders as {key:val,...} with `escape_keys=false`
    // (matches assistant-side convention — no <|"|> around keys at any
    // nesting level inside tool_call body).
    let body = format_tool_call_body(
        "draw",
        &json!({
            "point": {"x": 1, "y": 2}
        }),
    );
    assert!(
        body.contains("point:{x:1,y:2}") || body.contains("point:{y:2,x:1}"),
        "nested rendering wrong: {body}"
    );
}

#[test]
fn array_value() {
    let body = format_tool_call_body(
        "schedule",
        &json!({
            "days": ["mon", "wed", "fri"]
        }),
    );
    assert!(body.contains("days:[<|\"|>mon<|\"|>,<|\"|>wed<|\"|>,<|\"|>fri<|\"|>]"));
}

#[test]
fn rejects_empty_name_via_caller() {
    // The body formatter doesn't validate name (that's the caller's
    // contract — `encode_tool_call` in gemma4_chat enforces it). Confirm
    // the formatter doesn't *crash* on empty name; the surrounding
    // `<|tool_call>` markers would just wrap an unprintable body.
    let body = format_tool_call_body("", &json!({"k": "v"}));
    assert_eq!(body, "call:{k:<|\"|>v<|\"|>}");
}
