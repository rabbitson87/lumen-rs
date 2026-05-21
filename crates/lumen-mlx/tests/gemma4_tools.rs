//! Integration tests for `lumen_mlx::gemma4::render_tool_definitions_text`.
//!
//! Lives in `tests/` (not as an inline `#[cfg(test)] mod tests`) because the
//! library crate has pre-existing test-only compile errors in unrelated
//! modules (`gemma4_backend`, `turboquant`) that block `cargo test --lib`.
//! Integration tests compile against only the public API, so they remain
//! buildable / runnable while those broken `#[cfg(test)]` blocks exist.
//!
//! Coverage: the pure-text variant `render_tool_definitions_text` — this
//! pins the exact pseudo-JSON layout the model was trained against
//! (Gemma 4 canonical `format_function_declaration` jinja macro). The
//! tokenized variant `render_tool_definitions` requires a tokenizer.json
//! at runtime and is exercised at the engine wire-up step (Phase 1.3),
//! not here.

#![cfg(feature = "mlx-native")]

use lumen_mlx::gemma4::{ToolDef, render_tool_definitions_text};
use serde_json::json;

#[test]
fn minimal_tool_no_params() {
    let tools = [ToolDef {
        name: "ping",
        description: Some("Health check"),
        parameters: None,
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert_eq!(
        txt,
        "<|tool>declaration:ping{description:<|\"|>Health check<|\"|>}<tool|>"
    );
}

#[test]
fn single_required_string_param() {
    // OpenAI canonical example shape:
    //   get_weather(location: string, required)
    let params = json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    });
    let tools = [ToolDef {
        name: "get_weather",
        description: Some("Get current weather"),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    let expected = concat!(
        "<|tool>",
        "declaration:get_weather{",
        "description:<|\"|>Get current weather<|\"|>",
        ",parameters:{",
        "properties:{",
        "location:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>}",
        "},",
        "required:[<|\"|>location<|\"|>],",
        "type:<|\"|>OBJECT<|\"|>",
        "}",
        "}",
        "<tool|>",
    );
    assert_eq!(txt, expected);
}

#[test]
fn string_enum_param() {
    let params = json!({
        "type": "object",
        "properties": {
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": []
    });
    let tools = [ToolDef {
        name: "convert",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert!(
        txt.contains("enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>]"),
        "enum not rendered: {txt}"
    );
    assert!(txt.contains("type:<|\"|>STRING<|\"|>"));
}

#[test]
fn nested_object_property() {
    // Nested OBJECT — `address` field with its own properties + required.
    let params = json!({
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip":  {"type": "string"}
                },
                "required": ["city"]
            }
        },
        "required": ["address"]
    });
    let tools = [ToolDef {
        name: "lookup",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert!(txt.contains("address:{"));
    let inner = "properties:{city:{type:<|\"|>STRING<|\"|>},zip:{type:<|\"|>STRING<|\"|>}}";
    assert!(txt.contains(inner), "inner props missing in: {txt}");
    assert!(txt.contains("required:[<|\"|>city<|\"|>]"));
    assert!(txt.contains("type:<|\"|>OBJECT<|\"|>"));
}

#[test]
fn array_with_string_items() {
    let params = json!({
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": []
    });
    let tools = [ToolDef {
        name: "tag",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert!(
        txt.contains("items:{type:<|\"|>STRING<|\"|>}"),
        "items not rendered: {txt}"
    );
    assert!(txt.contains("type:<|\"|>ARRAY<|\"|>"));
}

#[test]
fn array_with_object_items() {
    // ARRAY whose items are themselves OBJECTs — exercises the
    // `properties` / `required` branches inside emit_items_body.
    let params = json!({
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string"},
                        "y": {"type": "string"}
                    },
                    "required": ["x", "y"]
                }
            }
        },
        "required": ["points"]
    });
    let tools = [ToolDef {
        name: "draw",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert!(
        txt.contains("properties:{x:{type:<|\"|>STRING<|\"|>},y:{type:<|\"|>STRING<|\"|>}}"),
        "items.properties missing: {txt}"
    );
    assert!(txt.contains("required:[<|\"|>x<|\"|>,<|\"|>y<|\"|>]"));
    assert!(txt.contains("type:<|\"|>OBJECT<|\"|>"));
    assert!(txt.contains("type:<|\"|>ARRAY<|\"|>"));
}

#[test]
fn nullable_property() {
    let params = json!({
        "type": "object",
        "properties": {
            "comment": {"type": "string", "nullable": true}
        },
        "required": []
    });
    let tools = [ToolDef {
        name: "review",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    assert!(txt.contains("nullable:true"), "nullable missing: {txt}");
}

#[test]
fn multiple_tools_in_order() {
    // Emission must preserve caller-supplied order (not sorted by
    // tool name) — clients rely on declaration order for some
    // routing heuristics.
    let p1 = json!({"type": "object", "properties": {}, "required": []});
    let p2 = json!({"type": "object", "properties": {}, "required": []});
    let tools = [
        ToolDef {
            name: "zeta",
            description: Some(""),
            parameters: Some(&p1),
            response: None,
        },
        ToolDef {
            name: "alpha",
            description: Some(""),
            parameters: Some(&p2),
            response: None,
        },
    ];
    let txt = render_tool_definitions_text(&tools).unwrap();
    let zeta_at = txt.find("declaration:zeta").unwrap();
    let alpha_at = txt.find("declaration:alpha").unwrap();
    assert!(
        zeta_at < alpha_at,
        "tool order not preserved (zeta should precede alpha): {txt}"
    );
}

#[test]
fn empty_tools_returns_empty_string() {
    let txt = render_tool_definitions_text(&[]).unwrap();
    assert!(txt.is_empty());
}

#[test]
fn rejects_empty_tool_name() {
    let params = json!({"type": "object", "properties": {}, "required": []});
    let tools = [ToolDef {
        name: "",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let err = render_tool_definitions_text(&tools).unwrap_err();
    assert!(err.to_string().contains("empty"));
}

#[test]
fn property_keys_sorted_alphabetically() {
    // Confirms `dictsort` behavior — properties emit in alpha order
    // regardless of input insertion order. Determinism matters for
    // KV-cache hits across requests with the same tool surface.
    let params = json!({
        "type": "object",
        "properties": {
            "zebra": {"type": "string"},
            "apple": {"type": "string"},
            "mango": {"type": "string"}
        },
        "required": []
    });
    let tools = [ToolDef {
        name: "x",
        description: Some(""),
        parameters: Some(&params),
        response: None,
    }];
    let txt = render_tool_definitions_text(&tools).unwrap();
    let a = txt.find("apple:").unwrap();
    let m = txt.find("mango:").unwrap();
    let z = txt.find("zebra:").unwrap();
    assert!(a < m && m < z, "keys not sorted: {txt}");
}
