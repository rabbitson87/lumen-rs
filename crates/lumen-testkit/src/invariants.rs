//! Contracts asserted by more than one driver.
//!
//! `crates/lumen-mlx/tests/fuzz_corpus_replay.rs` opens with a warning: each
//! replay body mirrors its fuzz target by hand, and drift between the two means
//! a committed crasher can pass replay while still crashing the soak — which
//! un-fixes the bug the corpus exists to pin. For the targets added here that
//! hazard is removed instead of documented: both drivers call the same
//! function, so there is nothing to keep in sync.
//!
//! Everything below takes values that were *already computed* by the caller,
//! so this crate needs no dependency on the crates under test.

use serde_json::Value;

/// The contract between `lumen_mlx::grammar::lark_literal` and `is_safe_ident`.
///
/// `lit` is `lark_literal(input)` and `safe_ident` is `is_safe_ident(input)`;
/// passing them in rather than calling them keeps this crate free of a
/// dependency on `lumen-mlx`.
///
/// Three properties, and the first is the one that matters. Tool names arrive
/// from clients, and a name that can close its own quote turns everything after
/// it in the emitted grammar into caller-supplied grammar rather than data —
/// `grammar-literal-escaping` shipped exactly there.
///
/// # Panics
/// On any violation; this is assertion machinery for tests and fuzz targets.
pub fn assert_lark_literal_contract(input: &str, lit: &str, safe_ident: bool) {
    let interior = lit
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or_else(|| panic!("literal is not quote-delimited: {lit:?} from {input:?}"));

    assert!(
        well_quoted(interior),
        "literal escaped its own quoting: {lit:?} from {input:?}"
    );

    // llguidance's Lark lexer rejects a raw ASCII control character inside a
    // literal, and a rejected grammar is a *dropped* grammar — the model is
    // then free to invent a tool nobody declared. That is `grammar-control-chars`,
    // which this assertion was added in response to and which the escape table
    // now covers.
    assert!(
        !interior.chars().any(|c| c.is_ascii_control()),
        "raw ASCII control character survived escaping: {lit:?} from {input:?}"
    );

    // Checked *in addition to* the quoting scan, not instead of it: a
    // `lark_literal` that had stopped escaping quotes altogether (`"` → `"""`)
    // still round-trips back to `"`, so a round trip alone would pass the
    // broken version.
    let back = unescape(interior)
        .unwrap_or_else(|| panic!("literal is not un-escapable: {lit:?} from {input:?}"));
    assert_eq!(back, input, "escape round trip lost information: {lit:?}");

    if safe_ident {
        assert_eq!(
            lit,
            format!("\"{input}\""),
            "is_safe_ident accepted a name that lark_literal had to escape"
        );
        assert!(
            input.is_ascii() && !input.is_empty(),
            "is_safe_ident accepted a name unusable as a Lark rule name: {input:?}"
        );
    }
}

/// The inverse of `lark_literal`'s escape table. `None` when the literal is not
/// something that table could have produced.
fn unescape(interior: &str) -> Option<String> {
    let mut out = String::with_capacity(interior.len());
    let mut chars = interior.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            '\\' => out.push('\\'),
            '"' => out.push('"'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            // `\uNNNN`, the spelling llguidance's Lark accepts for the ASCII
            // controls that have no shorthand. Exactly four hex digits.
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                if hex.len() != 4 {
                    return None;
                }
                let cp = u32::from_str_radix(&hex, 16).ok()?;
                out.push(char::from_u32(cp)?);
            }
            _ => return None,
        }
    }
    Some(out)
}

/// False when the interior breaks out of its quoting: a `"` no backslash
/// escapes, or a trailing lone `\` that would consume the closing quote.
fn well_quoted(interior: &str) -> bool {
    let mut escaped = false;
    for c in interior.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '"' => return false,
            _ => {}
        }
    }
    !escaped
}

/// The contract of `lumen_mlx::render_tools_system_block`: every declared tool
/// reaches the model, in order, and non-blank system content is not swallowed.
///
/// A tool dropped here fails invisibly — the model simply never calls it, which
/// reads as unhelpfulness rather than as a bug — so this compares the
/// *sequence* of names read back out of the rendered block against the declared
/// sequence. Duplicates are preserved by that comparison, and unlike a
/// substring count it is not fooled by a description that mentions another
/// tool's name, nor by a tool named `</tools>`.
///
/// # Panics
/// On any violation.
pub fn assert_tools_block_contract(declared: &[&str], rendered: &str, extra: Option<&str>) {
    // The opener precedes every tool's JSON, so the first hit is ours. The
    // closer is preceded by a raw newline, and in this region raw newlines
    // exist only as the per-tool separators — `serde_json` escapes any inside a
    // string — so the last such hit is ours even for a tool named `</tools>`.
    let open = rendered
        .find("<tools>")
        .unwrap_or_else(|| panic!("no <tools> opener in rendered block"))
        + "<tools>".len();
    let close = rendered
        .rfind("\n</tools>")
        .unwrap_or_else(|| panic!("no </tools> closer in rendered block"));
    let block = &rendered[open..close];

    let lines: Vec<&str> = if block.is_empty() {
        Vec::new()
    } else {
        block
            .strip_prefix('\n')
            .unwrap_or_else(|| panic!("tools block does not lead with a separator: {block:?}"))
            .split('\n')
            .collect()
    };

    assert_eq!(
        lines.len(),
        declared.len(),
        "tools block framing broke: {} lines for {} tools",
        lines.len(),
        declared.len()
    );

    // Owned, because the decoded name is not a substring of its own line
    // whenever it contained anything JSON escapes — a name holding a newline is
    // `\n` on the wire and a raw line feed after decoding.
    let seen: Vec<String> = lines
        .iter()
        .map(|line| {
            let v: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("tools line is not JSON ({e}): {line:?}"));
            v.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("tools line has no function.name: {line:?}"))
                .to_owned()
        })
        .collect();

    assert_eq!(seen, declared, "declared tools did not survive rendering");

    if let Some(trimmed) = extra.map(str::trim).filter(|t| !t.is_empty()) {
        assert!(
            rendered.contains(trimmed),
            "non-blank system content was dropped: {trimmed:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checker that accepts everything is worse than no checker, so each
    /// assertion is shown rejecting the shape it exists to catch.
    #[test]
    fn literal_contract_accepts_a_correct_escaping() {
        assert_lark_literal_contract("a\"b", "\"a\\\"b\"", false);
        assert_lark_literal_contract("plain", "\"plain\"", true);
        assert_lark_literal_contract("", "\"\"", false);
        assert_lark_literal_contract("a\nb", "\"a\\nb\"", false);
    }

    #[test]
    #[should_panic(expected = "escaped its own quoting")]
    fn literal_contract_rejects_an_unescaped_quote() {
        // What `lark_literal` would emit if it stopped escaping `"`. Note this
        // *does* round-trip, which is why the quoting scan is separate.
        assert_lark_literal_contract("a\"b", "\"a\"b\"", false);
    }

    #[test]
    #[should_panic(expected = "escaped its own quoting")]
    fn literal_contract_rejects_a_trailing_lone_backslash() {
        assert_lark_literal_contract("a\\", "\"a\\\"", false);
    }

    #[test]
    #[should_panic(expected = "round trip lost information")]
    fn literal_contract_rejects_a_dropped_character() {
        assert_lark_literal_contract("abc", "\"ab\"", false);
    }

    #[test]
    fn literal_contract_accepts_the_unicode_escape() {
        assert_lark_literal_contract("a\u{1b}b", "\"a\\u001bb\"", false);
        assert_lark_literal_contract("\u{0}", "\"\\u0000\"", false);
    }

    /// The shape the `grammar-control-chars` defect emitted: a raw ESC passed
    /// straight through, which round-trips perfectly and still makes
    /// llguidance's lexer reject the grammar.
    #[test]
    #[should_panic(expected = "raw ASCII control character survived")]
    fn literal_contract_rejects_a_raw_control_char() {
        assert_lark_literal_contract("a\u{1b}b", "\"a\u{1b}b\"", false);
    }

    #[test]
    #[should_panic(expected = "not un-escapable")]
    fn literal_contract_rejects_a_truncated_unicode_escape() {
        assert_lark_literal_contract("\u{1b}", "\"\\u01\"", false);
    }

    #[test]
    #[should_panic(expected = "had to escape")]
    fn literal_contract_rejects_a_safe_ident_that_needed_escaping() {
        assert_lark_literal_contract("a\"b", "\"a\\\"b\"", true);
    }

    fn render(names: &[&str]) -> String {
        let mut s = String::from("<|im_start|>system\n<tools>");
        for n in names {
            s.push('\n');
            s.push_str(
                &serde_json::to_string(&serde_json::json!({
                    "type": "function",
                    "function": { "name": n, "parameters": {} }
                }))
                .unwrap(),
            );
        }
        s.push_str("\n</tools>");
        s
    }

    #[test]
    fn tools_contract_accepts_a_faithful_render() {
        assert_tools_block_contract(&["a", "b"], &render(&["a", "b"]), None);
        assert_tools_block_contract(&[], &render(&[]), None);
        // Duplicates are legal input and must be preserved, not deduplicated.
        assert_tools_block_contract(&["a", "a"], &render(&["a", "a"]), None);
    }

    #[test]
    fn tools_contract_survives_a_tool_named_like_the_closer() {
        assert_tools_block_contract(&["</tools>"], &render(&["</tools>"]), None);
    }

    /// A name holding a raw newline is `\n` on the wire, so it is *not* a
    /// substring of its own rendered line. Comparing decoded names rather than
    /// searching the line is what makes this pass.
    #[test]
    fn tools_contract_survives_a_name_containing_a_newline() {
        assert_tools_block_contract(&["a\nb"], &render(&["a\nb"]), None);
    }

    #[test]
    #[should_panic(expected = "framing broke")]
    fn tools_contract_rejects_a_dropped_tool() {
        assert_tools_block_contract(&["a", "b"], &render(&["a"]), None);
    }

    #[test]
    #[should_panic(expected = "did not survive rendering")]
    fn tools_contract_rejects_a_reordered_render() {
        assert_tools_block_contract(&["a", "b"], &render(&["b", "a"]), None);
    }

    #[test]
    #[should_panic(expected = "system content was dropped")]
    fn tools_contract_rejects_swallowed_system_content() {
        assert_tools_block_contract(&["a"], &render(&["a"]), Some("  keep me  "));
    }

    #[test]
    fn tools_contract_ignores_blank_system_content() {
        assert_tools_block_contract(&["a"], &render(&["a"]), Some("   \n  "));
    }
}
