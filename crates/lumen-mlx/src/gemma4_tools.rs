//! Gemma 4 tool-definition renderer.
//!
//! Ports the `format_function_declaration` / `format_parameters` /
//! `format_argument` macros from the canonical Gemma 4 `chat_template.jinja`
//! to Rust. Produces the token sequence the model was trained against for
//! tool-aware system headers:
//!
//! ```text
//! <|tool>declaration:NAME{
//!     description:<|"|>DESC<|"|>,
//!     parameters:{
//!         properties:{
//!             KEY:{description:<|"|>D<|"|>,type:<|"|>STRING<|"|>},
//!             ...
//!         },
//!         required:[<|"|>k1<|"|>,<|"|>k2<|"|>],
//!         type:<|"|>OBJECT<|"|>
//!     }
//! }<tool|>
//! ```
//!
//! Wire-format rules (faithful to the jinja):
//!   - **Keys** are emitted unquoted by default (`description:`, `type:`,
//!     `parameters:`). Inside `enum:[...]` and other top-level argument
//!     positions where the user might supply structured data, string keys
//!     are escaped with `<|"|>`-pairs.
//!   - **String values** are always escaped with `<|"|>`-pairs.
//!   - **Numbers / booleans / arrays** are emitted raw.
//!   - **JSON Schema `type`** is upper-cased (`"string"` → `STRING`).
//!   - Property maps are sorted by key (matches jinja `| dictsort`) so the
//!     output is deterministic regardless of `serde_json::Value` insertion
//!     order.
//!
//! Encoded as a single `encode_plain(...)` call — the Gemma 4 tokenizer's
//! `added_tokens` table maps `<|tool>` / `<tool|>` / `<|"|>` to their
//! single-token IDs at encode time, so we don't need to splice constants
//! manually like `render_tool_response_block` does. Mid-text special
//! tokens are still produced as the correct single IDs because the
//! tokenizer regex-matches them before the BPE step.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // re-exported via gemma4::tools once the server wires it
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::fmt::Write;

    use crate::gemma4_chat::imp::Gemma4ChatTemplate;

    // The `ToolDef` struct lives in non-feature-gated `chat_io` so
    // lumen-server can hold it without depending on `mlx-native`. The
    // renderers below stay feature-gated because they call into the
    // Gemma 4 tokenizer.
    pub use crate::chat_io::ToolDef;

    /// Render the tool-definition section that goes inside the system turn.
    ///
    /// Emits one `<|tool>declaration:NAME{...}<tool|>` block per tool, in
    /// the order supplied. Returns a flat token id sequence the caller can
    /// concatenate with the rest of the system prompt.
    ///
    /// Caller contract: the result is meant to live *between* the system
    /// header text and the `<turn|>` close of a `<|turn>system\n...` block,
    /// matching the canonical chat_template.jinja layout.
    pub fn render_tool_definitions(
        template: &Gemma4ChatTemplate,
        tools: &[ToolDef<'_>],
    ) -> Result<Vec<u32>> {
        if tools.is_empty() {
            return Ok(Vec::new());
        }
        let mut text = String::new();
        for (i, tool) in tools.iter().enumerate() {
            text.push_str("<|tool>");
            format_function_declaration(&mut text, tool)
                .with_context(|| format!("format_function_declaration tool[{i}] {:?}", tool.name))?;
            text.push_str("<tool|>");
        }
        template
            .encode_plain(&text)
            .context("encode rendered tool definitions")
    }

    // ─────────────────────── pseudo-JSON serializers ───────────────────────

    /// Standard JSON Schema keys that get special handling inside nested
    /// OBJECT properties (mirrors jinja's `standard_keys`). When
    /// `filter_keys=true` (inline-object case), these are emitted by the
    /// dedicated branches and skipped from the generic loop.
    const STANDARD_KEYS: &[&str] = &[
        "description",
        "type",
        "properties",
        "required",
        "nullable",
    ];

    /// `format_function_declaration` macro port:
    ///
    /// ```text
    /// declaration:NAME{description:<|"|>D<|"|>,parameters:{...},response:{...}}
    /// ```
    fn format_function_declaration(out: &mut String, tool: &ToolDef<'_>) -> Result<()> {
        if tool.name.is_empty() {
            return Err(anyhow!("ToolDef::name is empty"));
        }
        write!(out, "declaration:{}{{", tool.name).expect("write to String never fails");

        // description:<|"|>...<|"|>  — emitted unconditionally because the
        // jinja `format_function_declaration` always emits it (the field is
        // pre-trimmed by the caller, an empty string is still wrapped).
        if let Some(desc) = tool.description {
            out.push_str("description:<|\"|>");
            out.push_str(desc);
            out.push_str("<|\"|>");
        } else {
            // Match jinja's behavior when description is missing: emit the
            // empty form so the prompt shape stays consistent with the
            // training distribution.
            out.push_str("description:<|\"|><|\"|>");
        }

        // parameters:{...}
        if let Some(params) = tool.parameters {
            out.push(',');
            out.push_str("parameters:{");
            let properties = params.get("properties");
            let required = params
                .get("required")
                .and_then(|v| v.as_array())
                .map(|a| a.as_slice());
            if let Some(props) = properties.and_then(|p| p.as_object()) {
                out.push_str("properties:{");
                format_parameters(out, props, required, /* filter_keys */ false)?;
                out.push_str("},");
            }
            if let Some(req) = required {
                out.push_str("required:[");
                emit_string_array(out, req);
                out.push_str("],");
            }
            if let Some(ty) = params.get("type").and_then(|t| t.as_str()) {
                out.push_str("type:<|\"|>");
                out.push_str(&ty.to_uppercase());
                out.push_str("<|\"|>");
            }
            out.push('}');
        }

        // response:{...} — optional
        if let Some(resp) = tool.response {
            out.push(',');
            out.push_str("response:{");
            if let Some(desc) = resp.get("description").and_then(|d| d.as_str()) {
                out.push_str("description:<|\"|>");
                out.push_str(desc);
                out.push_str("<|\"|>,");
            }
            if let Some(ty) = resp.get("type").and_then(|t| t.as_str()) {
                if ty.eq_ignore_ascii_case("object") {
                    out.push_str("type:<|\"|>");
                    out.push_str(&ty.to_uppercase());
                    out.push_str("<|\"|>");
                }
            }
            out.push('}');
        }
        out.push('}');
        Ok(())
    }

    /// `format_parameters` macro port. Iterates `properties` in sorted key
    /// order (jinja `| dictsort`).
    fn format_parameters(
        out: &mut String,
        properties: &serde_json::Map<String, Value>,
        _required: Option<&[Value]>,
        filter_keys: bool,
    ) -> Result<()> {
        // BTreeMap sorts by key for deterministic emit order.
        let sorted: BTreeMap<&str, &Value> =
            properties.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut first = true;
        for (key, value) in sorted {
            if filter_keys && STANDARD_KEYS.contains(&key) {
                continue;
            }
            if !first {
                out.push(',');
            }
            first = false;
            format_property(out, key, value)?;
        }
        Ok(())
    }

    /// Emit a single property entry: `KEY:{description:..., type:..., ...}`.
    fn format_property(out: &mut String, key: &str, value: &Value) -> Result<()> {
        out.push_str(key);
        out.push_str(":{");

        let mut emitted = false;

        if let Some(desc) = value.get("description").and_then(|d| d.as_str()) {
            out.push_str("description:<|\"|>");
            out.push_str(desc);
            out.push_str("<|\"|>");
            emitted = true;
        }

        let type_upper = value
            .get("type")
            .and_then(|t| t.as_str())
            .map(|s| s.to_uppercase());

        match type_upper.as_deref() {
            Some("STRING") => {
                if let Some(enum_vals) = value.get("enum").and_then(|e| e.as_array()) {
                    if emitted {
                        out.push(',');
                    }
                    out.push_str("enum:");
                    format_argument_array(out, enum_vals, /* escape_keys */ true);
                    emitted = true;
                }
            }
            Some("ARRAY") => {
                if let Some(items) = value.get("items").and_then(|i| i.as_object()) {
                    if emitted {
                        out.push(',');
                    }
                    out.push_str("items:{");
                    emit_items_body(out, items)?;
                    out.push('}');
                    emitted = true;
                }
            }
            _ => {}
        }

        if value
            .get("nullable")
            .and_then(|n| n.as_bool())
            .unwrap_or(false)
        {
            if emitted {
                out.push(',');
            }
            out.push_str("nullable:true");
            emitted = true;
        }

        if type_upper.as_deref() == Some("OBJECT") {
            // Two shapes the jinja handles:
            //   (a) value.properties is a mapping → recurse into it,
            //       using value.required for required-list rendering.
            //   (b) value has no .properties but value itself is mapping
            //       → treat value as an inline-properties dict, filtering
            //       standard keys (description/type/required/nullable).
            if let Some(props) = value.get("properties").and_then(|p| p.as_object()) {
                if emitted {
                    out.push(',');
                }
                out.push_str("properties:{");
                let req = value
                    .get("required")
                    .and_then(|v| v.as_array())
                    .map(|a| a.as_slice());
                format_parameters(out, props, req, /* filter_keys */ false)?;
                out.push('}');
                emitted = true;
            } else if let Some(inline) = value.as_object() {
                if emitted {
                    out.push(',');
                }
                out.push_str("properties:{");
                let req = value
                    .get("required")
                    .and_then(|v| v.as_array())
                    .map(|a| a.as_slice());
                format_parameters(out, inline, req, /* filter_keys */ true)?;
                out.push('}');
                emitted = true;
            }

            if let Some(req) = value.get("required").and_then(|r| r.as_array()) {
                if !req.is_empty() {
                    if emitted {
                        out.push(',');
                    }
                    out.push_str("required:[");
                    emit_string_array(out, req);
                    out.push(']');
                    emitted = true;
                }
            }
        }

        // type:<|"|>TYPE<|"|>  always emitted last
        if emitted {
            out.push(',');
        }
        out.push_str("type:<|\"|>");
        out.push_str(type_upper.as_deref().unwrap_or("STRING"));
        out.push_str("<|\"|>}");
        Ok(())
    }

    /// Body of an ARRAY's `items:{...}` block. Mirrors the jinja branches
    /// over `(item_key, item_value)` pairs.
    fn emit_items_body(out: &mut String, items: &serde_json::Map<String, Value>) -> Result<()> {
        let sorted: BTreeMap<&str, &Value> =
            items.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut first = true;
        for (key, value) in sorted {
            if value.is_null() {
                continue;
            }
            if !first {
                out.push(',');
            }
            first = false;
            match key {
                "properties" => {
                    out.push_str("properties:{");
                    if let Some(map) = value.as_object() {
                        let req = items
                            .get("required")
                            .and_then(|r| r.as_array())
                            .map(|a| a.as_slice());
                        format_parameters(out, map, req, /* filter_keys */ false)?;
                    }
                    out.push('}');
                }
                "required" => {
                    if let Some(arr) = value.as_array() {
                        out.push_str("required:[");
                        emit_string_array(out, arr);
                        out.push(']');
                    }
                }
                "type" => {
                    out.push_str("type:");
                    if let Some(ty) = value.as_str() {
                        out.push_str("<|\"|>");
                        out.push_str(&ty.to_uppercase());
                        out.push_str("<|\"|>");
                    } else if let Some(types) = value.as_array() {
                        // type can be an array of strings (JSON Schema
                        // union); jinja maps each to upper and emits as a
                        // string array.
                        let upper: Vec<Value> = types
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| Value::String(s.to_uppercase())))
                            .collect();
                        format_argument_array(out, &upper, /* escape_keys */ true);
                    }
                }
                _ => {
                    out.push_str(key);
                    out.push(':');
                    format_argument(out, value, /* escape_keys */ true);
                }
            }
        }
        Ok(())
    }

    /// `format_argument(arg, escape_keys)` macro port.
    fn format_argument(out: &mut String, value: &Value, escape_keys: bool) {
        match value {
            Value::String(s) => {
                out.push_str("<|\"|>");
                out.push_str(s);
                out.push_str("<|\"|>");
            }
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Null => out.push_str("null"),
            Value::Number(n) => write!(out, "{n}").expect("number to String"),
            Value::Array(arr) => format_argument_array(out, arr, escape_keys),
            Value::Object(map) => {
                out.push('{');
                let sorted: BTreeMap<&str, &Value> =
                    map.iter().map(|(k, v)| (k.as_str(), v)).collect();
                let mut first = true;
                for (key, v) in sorted {
                    if !first {
                        out.push(',');
                    }
                    first = false;
                    if escape_keys {
                        out.push_str("<|\"|>");
                        out.push_str(key);
                        out.push_str("<|\"|>");
                    } else {
                        out.push_str(key);
                    }
                    out.push(':');
                    format_argument(out, v, escape_keys);
                }
                out.push('}');
            }
        }
    }

    fn format_argument_array(out: &mut String, arr: &[Value], escape_keys: bool) {
        out.push('[');
        for (i, item) in arr.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            format_argument(out, item, escape_keys);
        }
        out.push(']');
    }

    /// Emit a string array as `[<|"|>v1<|"|>,<|"|>v2<|"|>,...]` (used for
    /// `required:[...]`).
    fn emit_string_array(out: &mut String, arr: &[Value]) {
        let mut first = true;
        for item in arr {
            if let Some(s) = item.as_str() {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str("<|\"|>");
                out.push_str(s);
                out.push_str("<|\"|>");
            }
        }
    }

    // ─────────────────────── pure-text fixture helper ──────────────────────

    /// Render tool definitions to the plain-text form (without tokenization).
    /// Useful for golden-text fixtures that pin the exact pseudo-JSON layout
    /// without requiring a tokenizer.json. The tokenized form is what the
    /// model sees, but the text form is what mlx_lm.apply_chat_template
    /// produces internally before tokenization.
    pub fn render_tool_definitions_text(tools: &[ToolDef<'_>]) -> Result<String> {
        let mut text = String::new();
        for tool in tools {
            text.push_str("<|tool>");
            format_function_declaration(&mut text, tool)?;
            text.push_str("<tool|>");
        }
        Ok(text)
    }

    // Unit tests live in `crates/lumen-mlx/tests/gemma4_tools.rs` as
    // integration tests so they don't get entangled with pre-existing
    // compile errors in the library's `#[cfg(test)]` blocks for unrelated
    // modules (gemma4_backend, turboquant). Integration tests compile
    // against the public API only and run cleanly.
}
