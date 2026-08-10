//! Gemma 4 tool-call *syntax* — the pure text half of `gemma4_response`.
//!
//! Parses the pseudo-JSON a Gemma 4 model emits between `<|tool_call>` (48)
//! and `<tool_call|>` (49):
//!
//! ```text
//! call:NAME{key:<|"|>value<|"|>,other:7}
//! ```
//!
//! Reference: `mlx_lm/tool_parsers/gemma4.py`.
//!
//! ## Why this is its own module
//!
//! Everything here is `&str` in, `serde_json::Value` out. No tokenizer, no
//! `mlx_rs`, no GPU — so unlike the streaming [`crate::gemma4_response`]
//! parser it wraps (which needs a tokenizer-backed `Gemma4ChatTemplate` and is
//! therefore `#[cfg(feature = "mlx-native")]`), it can be tested, fuzzed and
//! measured for coverage under the crate's plain `default = []` build.
//!
//! That matters more than tidiness. Of the defects recorded in
//! `xtask/src/red_green.rs`, several come from exactly these few hundred lines
//! — `tool-name-scanner` (`call:bad call:good{x:1}` parsed as one tool named
//! `"bad call:good"`, a name no client declared) and `args-unicode-keys`
//! (`{도시:…}` arriving as mojibake). Both are reachable from a hostile string
//! alone, which is the cheapest possible thing to test, and both shipped
//! anyway because the code sat behind a feature gate that the fast test path
//! never turned on. [`crate::chat_io`] exists for the same reason on the
//! data-type side.

use anyhow::{Context, Result, anyhow};
use serde_json::Value as JsonValue;

use crate::chat_io::ParsedToolCall;

/// Parse `call:NAME{...}` (potentially multiple such blocks) out of a
/// decoded `<|tool_call>…<tool_call|>` body.
///
/// Mirrors `mlx_lm.tool_parsers.gemma4.parse_tool_call` minus the
/// recursive `(?R)` regex — we do balanced-brace matching by hand
/// because the standard `regex` crate doesn't support recursion.
pub fn parse_tool_call_body(text: &str) -> Result<Vec<ParsedToolCall>> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Find next "call:"
        let Some(call_pos) = find_substr(text, "call:", i) else {
            break;
        };
        // Read NAME permissively — any character up to the first
        // '{'. Accepts non-OpenAI-spec function names (spaces, parens,
        // dots, …) so clients like Ayla can pass MCP-prefixed names
        // through without sanitizing — e.g.
        // `Playwright (Stealth)__browser_navigate`.
        //
        // Three hard boundaries stop the scan. A newline or carriage
        // return, because a name never spans lines. And a further
        // `call:`, because that is the opener of the *next* call: without
        // it, a malformed `call:no_brace call:good{x:1}` swallows the
        // second opener into the first name and yields a tool named
        // "no_brace call:good" — a name no client declared, silently
        // wrong rather than loudly skipped. Whitespace deliberately does
        // NOT stop the scan; that is what makes the MCP names above work.
        let name_start = call_pos + "call:".len();
        let mut brace_start = name_start;
        let mut hit_next_call = false;
        while brace_start < bytes.len() {
            let b = bytes[brace_start];
            if b == b'{' || b == b'\n' || b == b'\r' {
                break;
            }
            // Compared as bytes: `brace_start` walks byte-by-byte and a
            // name may be non-ASCII, so slicing `text` here could land
            // mid-character. "call:" is pure ASCII and UTF-8 continuation
            // bytes never collide with it, so a byte match implies a real
            // character boundary.
            if bytes[brace_start..].starts_with(b"call:") {
                hit_next_call = true;
                break;
            }
            brace_start += 1;
        }
        if brace_start >= bytes.len() || bytes[brace_start] != b'{' {
            // Malformed: this opener has no args block. Resume at the
            // next `call:` when that is what stopped us — the outer
            // search then re-finds it and parses it properly. Otherwise
            // step past this opener so a later one can still be found.
            // Either way `i` strictly advances, so the loop terminates.
            i = if hit_next_call {
                brace_start
            } else {
                name_start
            };
            continue;
        }
        // Trim trailing whitespace from NAME.
        let mut name_end = brace_start;
        while name_end > name_start && bytes[name_end - 1].is_ascii_whitespace() {
            name_end -= 1;
        }
        if name_end == name_start {
            i = brace_start + 1;
            continue;
        }
        let name = &text[name_start..name_end];

        // Balanced-brace span starting at brace_start.
        let brace_end = match_balanced_braces(text, brace_start)
            .ok_or_else(|| anyhow!("tool-call: unbalanced braces near {brace_start}"))?;
        let args_raw = &text[brace_start..=brace_end];
        let arguments = gemma4_args_to_json(args_raw)
            .with_context(|| format!("tool-call '{name}': arg→json"))?;
        out.push(ParsedToolCall {
            name: name.to_string(),
            arguments,
        });
        i = brace_end + 1;
    }
    if out.is_empty() {
        return Err(anyhow!("tool-call: no call:NAME{{…}} found in body"));
    }
    Ok(out)
}

/// Convert Gemma 4 tool-call argument syntax (`{key:<|"|>val<|"|>,...}`)
/// into a strict JSON value.
///
/// Algorithm mirrors `mlx_lm.tool_parsers.gemma4._gemma4_args_to_json`:
///   1. Replace every `<|"|>...<|"|>` literal with a placeholder.
///   2. Quote bare keys (`,key:` → `,"key":` and `{key:` → `{"key":`).
///   3. Substitute placeholders back as JSON-escaped string literals.
///   4. Parse as JSON.
pub fn gemma4_args_to_json(text: &str) -> Result<JsonValue> {
    const STR_DELIM: &str = "<|\"|>";

    // extract strings, replace with placeholders.
    let mut strings: Vec<String> = Vec::new();
    let mut buf = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let Some(open) = rest.find(STR_DELIM) else {
            buf.push_str(rest);
            break;
        };
        buf.push_str(&rest[..open]);
        let after_open = &rest[open + STR_DELIM.len()..];
        let Some(close) = after_open.find(STR_DELIM) else {
            return Err(anyhow!("unterminated <|\"|> string literal"));
        };
        let s = &after_open[..close];
        strings.push(s.to_string());
        buf.push_str(&format!("\x00{}\x00", strings.len() - 1));
        rest = &after_open[close + STR_DELIM.len()..];
    }

    // Quote bare keys. A bare key follows '{' or ',' (after optional
    // whitespace) and is terminated by ':'.
    //
    // Both this pass and the substitution below walk `&str` rather than
    // bytes. They used to index bytes and rebuild with `b as char`, which
    // is a Latin-1 reinterpretation: every byte of a multi-byte character
    // became its own `char` and was re-encoded, so `{도시:…}` came out as
    // `{Ã«Â\u{8f}Â\u{84}…}` and the JSON parse failed. Only text that had
    // already been lifted into `strings` survived, which is why a non-ASCII
    // *value* inside `<|"|>` worked while a non-ASCII *key* did not.
    //
    // Identifier characters are Unicode-alphanumeric for the same reason
    // the name scanner in `parse_tool_call_body` is permissive: the model
    // emits whatever key the tool schema declared, and a schema is free to
    // declare `도시`. An ASCII-only class would leave such a key unquoted
    // and produce invalid JSON even with the encoding fixed.
    let mut quoted = String::with_capacity(buf.len() + 16);
    let mut rest = buf.as_str();
    loop {
        let Some(pos) = rest.find(['{', ',']) else {
            quoted.push_str(rest);
            break;
        };
        quoted.push_str(&rest[..=pos]); // '{' and ',' are ASCII: safe slice
        let after = &rest[pos + 1..];
        let ws_end = after
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(after.len());
        quoted.push_str(&after[..ws_end]);
        let ident_area = &after[ws_end..];
        let id_end = ident_area
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
            .unwrap_or(ident_area.len());
        if id_end > 0 && ident_area[id_end..].starts_with(':') {
            quoted.push('"');
            quoted.push_str(&ident_area[..id_end]);
            quoted.push('"');
            rest = &ident_area[id_end..]; // resumes on the ':'
        } else {
            rest = ident_area;
        }
    }

    // Substitute placeholders with JSON-escaped strings. The markers are
    // NUL, which is single-byte, so every slice below lands on a character
    // boundary and the text between them passes through untouched.
    let mut final_str = String::with_capacity(quoted.len() + 32);
    let mut rest = quoted.as_str();
    while let Some(open) = rest.find('\0') {
        final_str.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('\0') else {
            return Err(anyhow!("placeholder NUL mismatch"));
        };
        let n: usize = after[..close].parse().context("placeholder index")?;
        let s = strings.get(n).ok_or_else(|| anyhow!("placeholder oob"))?;
        final_str.push_str(&serde_json::to_string(s).context("escape string")?);
        rest = &after[close + 1..];
    }
    final_str.push_str(rest);

    serde_json::from_str(&final_str)
        .with_context(|| format!("tool-call args JSON parse: {final_str:?}"))
}

fn find_substr(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    haystack[from..].find(needle).map(|p| p + from)
}

fn match_balanced_braces(text: &str, open_at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if open_at >= bytes.len() || bytes[open_at] != b'{' {
        return None;
    }
    const STR_DELIM_BYTES: &[u8] = b"<|\"|>";
    let mut depth: i32 = 0;
    let mut i = open_at;
    while i < bytes.len() {
        // Skip <|"|>…<|"|> string literals (braces inside are not
        // structural).
        if bytes[i..].starts_with(STR_DELIM_BYTES) {
            let after = i + STR_DELIM_BYTES.len();
            let rest = &bytes[after..];
            if let Some(pos) = rest
                .windows(STR_DELIM_BYTES.len())
                .position(|w| w == STR_DELIM_BYTES)
            {
                i = after + pos + STR_DELIM_BYTES.len();
                continue;
            } else {
                return None;
            }
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

// ───────────────────────── tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_parser_simple_string_value() {
        let v = gemma4_args_to_json(r#"{name:<|"|>Tokyo<|"|>}"#).expect("parse");
        assert_eq!(v["name"], "Tokyo");
    }

    #[test]
    fn arg_parser_multiple_fields_mixed_types() {
        let v = gemma4_args_to_json(r#"{city:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>,days:7}"#)
            .expect("parse");
        assert_eq!(v["city"], "Paris");
        assert_eq!(v["unit"], "celsius");
        assert_eq!(v["days"], 7);
    }

    #[test]
    fn arg_parser_nested_object() {
        let v = gemma4_args_to_json(r#"{location:{lat:35.6,lng:139.7,name:<|"|>Tokyo<|"|>}}"#)
            .expect("parse");
        assert_eq!(v["location"]["lat"], 35.6);
        assert_eq!(v["location"]["lng"], 139.7);
        assert_eq!(v["location"]["name"], "Tokyo");
    }

    #[test]
    fn arg_parser_array_of_strings() {
        let v = gemma4_args_to_json(r#"{tags:[<|"|>red<|"|>,<|"|>blue<|"|>]}"#).expect("parse");
        assert!(v["tags"].is_array());
        assert_eq!(v["tags"][0], "red");
        assert_eq!(v["tags"][1], "blue");
    }

    #[test]
    fn arg_parser_string_with_braces_and_commas() {
        // Internal characters that look like JSON structure must NOT
        // be misread (they live inside a <|"|>...<|"|> literal).
        let v = gemma4_args_to_json(r#"{q:<|"|>hello, {world}<|"|>}"#).expect("parse");
        assert_eq!(v["q"], "hello, {world}");
    }

    #[test]
    fn arg_parser_unterminated_string_errors() {
        let err = gemma4_args_to_json(r#"{q:<|"|>broken}"#).unwrap_err();
        assert!(format!("{err}").contains("unterminated"), "got: {err}");
    }

    #[test]
    fn body_parser_single_call() {
        let body = r#"call:get_weather{city:<|"|>Seoul<|"|>}"#;
        let calls = parse_tool_call_body(body).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Seoul");
    }

    #[test]
    fn body_parser_multiple_calls() {
        let body = r#"call:a{x:1}call:b{y:<|"|>z<|"|>}"#;
        let calls = parse_tool_call_body(body).expect("parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments["x"], 1);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments["y"], "z");
    }

    #[test]
    fn body_parser_rejects_empty_body() {
        let err = parse_tool_call_body("(no calls here)").unwrap_err();
        assert!(format!("{err}").contains("no call:"));
    }

    #[test]
    fn body_parser_accepts_spaces_and_parens_in_name() {
        // Ayla MCP server prefix has spaces+parens — must round-trip
        // without 500 error. Mirrors `getAllTools()` output shape.
        let body =
            r#"call:Playwright (Stealth)__browser_navigate{url:<|"|>https://example.com<|"|>}"#;
        let calls = parse_tool_call_body(body).expect("parse permissive name");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Playwright (Stealth)__browser_navigate");
        assert_eq!(calls[0].arguments["url"], "https://example.com");
    }

    #[test]
    fn body_parser_stops_a_name_at_the_next_opener() {
        // The counterweight to the permissive name above. A `call:`
        // without an args block must not swallow the opener that follows
        // it: the greedy scan used to yield one call named
        // "bad_no_brace call:good", which no client ever declared, and a
        // fuzzy name matcher downstream can turn that into a *plausible*
        // wrong tool. Skipping the malformed opener and parsing the real
        // call is the only reading that cannot invent a tool.
        let calls = parse_tool_call_body("call:bad_no_brace call:good{x:1}")
            .expect("recovers to the well-formed call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "good");
        assert_eq!(calls[0].arguments["x"], 1);
    }

    #[test]
    fn body_parser_skips_a_run_of_malformed_openers() {
        // Several bad openers in a row must not stall the scan — each one
        // has to advance it — and must not merge into one another.
        let calls = parse_tool_call_body("call:a call:b call:c{x:1}")
            .expect("recovers past every malformed opener");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "c");
    }

    #[test]
    fn body_parser_reports_no_call_when_every_opener_is_malformed() {
        // No args block anywhere: an error, not a phantom tool call, and
        // not a hang.
        let err =
            parse_tool_call_body("call:a call:b call:c").expect_err("nothing well-formed to parse");
        assert!(format!("{err}").contains("no call:"), "{err}");
    }

    #[test]
    fn body_parser_stops_a_non_ascii_name_at_the_next_opener() {
        // The scan walks bytes, so a multi-byte name is where an
        // ASCII-only boundary check would slice mid-character and panic.
        let calls = parse_tool_call_body("call:날씨_조회 call:good{x:1}")
            .expect("recovers past a non-ASCII malformed opener");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "good");
    }

    #[test]
    fn body_parser_keeps_a_non_ascii_name_that_is_well_formed() {
        let calls = parse_tool_call_body(r#"call:날씨_조회{city:<|"|>서울<|"|>}"#)
            .expect("non-ASCII names still round-trip");
        assert_eq!(calls[0].name, "날씨_조회");
        assert_eq!(calls[0].arguments["city"], "서울");
    }

    #[test]
    fn args_to_json_quotes_a_non_ascii_bare_key() {
        // A tool schema may declare any key it likes. The quoter used to
        // rebuild the body with `byte as char`, turning `도시` into
        // Latin-1 mojibake and failing the JSON parse; an ASCII-only
        // identifier class would also have left the key unquoted.
        let v =
            gemma4_args_to_json(r#"{도시:<|"|>서울<|"|>,count:2}"#).expect("non-ASCII bare key");
        assert_eq!(v["도시"], "서울");
        assert_eq!(v["count"], 2);
    }

    #[test]
    fn args_to_json_preserves_non_ascii_outside_string_literals() {
        // Bare (unwrapped) non-ASCII values never entered the placeholder
        // table, so they went through the byte→char rebuild unprotected.
        // `null`-style bare tokens are the realistic case; this pins that
        // nothing outside `<|"|>` gets re-encoded.
        let v = gemma4_args_to_json(r#"{"도시":"서울","n":null}"#)
            .expect("already-quoted non-ASCII passes through");
        assert_eq!(v["도시"], "서울");
        assert!(v["n"].is_null());
    }

    #[test]
    fn args_to_json_handles_non_ascii_in_nested_and_array_positions() {
        let v = gemma4_args_to_json(
            r#"{요청:{도시:<|"|>서울<|"|>,태그:[<|"|>맑음<|"|>,<|"|>바람<|"|>]}}"#,
        )
        .expect("nested non-ASCII keys");
        assert_eq!(v["요청"]["도시"], "서울");
        assert_eq!(v["요청"]["태그"][1], "바람");
    }

    #[test]
    fn args_to_json_leaves_array_elements_unquoted() {
        // The quoter only fires on `{`/`,` followed by ident + ':'. An
        // array element after a comma is ident-shaped but has no colon, so
        // it must pass through — a regression here would corrupt every
        // list argument.
        let v = gemma4_args_to_json(r#"{flags:[true,false],ns:[1,2]}"#).expect("arrays");
        assert_eq!(v["flags"][0], true);
        assert_eq!(v["ns"][1], 2);
    }

    #[test]
    fn body_parser_trims_trailing_whitespace_from_name() {
        // Whitespace between NAME and `{` must be tolerated AND
        // stripped from the emitted name.
        let body = r#"call:my tool   {x:1}"#;
        let calls = parse_tool_call_body(body).expect("parse trimmed name");
        assert_eq!(calls[0].name, "my tool");
    }

    #[test]
    fn body_parser_handles_dots_in_name() {
        // Some MCP / non-OpenAI clients use dots (e.g. namespace.method).
        let body = r#"call:fs.read_file{path:<|"|>/tmp/x<|"|>}"#;
        let calls = parse_tool_call_body(body).expect("parse dotted name");
        assert_eq!(calls[0].name, "fs.read_file");
    }
}
