//! Gemma 4 streaming response parser.
//!
//! Two responsibilities, both consumed token-by-token from `generate()`:
//!
//! 1. **Reasoning channel demux** — Gemma 4 separates the model's chain-of-
//!    thought from its visible reply via `<|channel>thought\n…<channel|>`
//!    blocks. Tokens received between the open/close special tokens belong
//!    to the `reasoning` field of the OpenAI-style response; everything
//!    else is visible content.
//!
//! 2. **Tool call extraction** — function calls land between special tokens
//!    `<|tool_call>` (48) and `<tool_call|>` (49) using Gemma 4's own
//!    pseudo-JSON syntax (`call:NAME{key:<|"|>val<|"|>,...}`). The parser
//!    converts that into a `serde_json::Value` so the HTTP layer can ship
//!    OpenAI-compatible `tool_calls[]`.
//!
//! Reference: `mlx_lm/tool_parsers/gemma4.py` + `chat_template.jinja`.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // surfaced via gemma4_response::imp::* once the server lands
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use serde_json::Value as JsonValue;

    use crate::gemma4_chat::imp::{Gemma4ChatTemplate, TOK_CHANNEL_CLOSE, TOK_CHANNEL_OPEN};

    // Re-export the non-feature-gated data types from `chat_io` so existing
    // call sites (`gemma4_response::imp::ParsedResponse` etc.) keep
    // resolving. The parser itself stays feature-gated since it depends on
    // the tokenizer-backed `Gemma4ChatTemplate`.
    pub use crate::chat_io::{ParsedResponse, ParsedToolCall};

    // Tool-call delimiters from `tokenizer.json` added_tokens:
    //   <|tool_call> = 48,  <tool_call|> = 49
    pub const TOK_TOOL_CALL_OPEN: u32 = 48;
    pub const TOK_TOOL_CALL_CLOSE: u32 = 49;

    // NOTE — plain-text `<think>…</think>` is deliberately NOT demuxed.
    //
    // Some Gemma 4 `it` builds (e.g. `mlx-community/gemma-4-26b-a4b-it-nvfp4`,
    // the weights Ollama serves) wrap chain-of-thought in the literal text
    // markers `<think>…</think>` rather than the `<|channel>`(100)/`<channel|>`(101)
    // special tokens. We intentionally let those flow through as **visible
    // content**, byte-for-byte matching Ollama's gemma4 parser (which only
    // recognizes the `<|channel>` special tokens — see
    // `ollama/model/parsers/gemma4.go`). Reasons:
    //   1. Streaming parity — Ollama streams `<think>…` as content deltas;
    //      demuxing it into a separate `reasoning` field diverges from what
    //      text-tag clients (Ayla `ChatWindow.tsx`) parse out of content.
    //   2. Robustness — when the model rambles and hits EOS *before* emitting
    //      a closing `</think>`, demuxing strands the whole reply in
    //      `reasoning` (empty visible → blank bubble). Leaving it in content
    //      means the user always sees the model's output, exactly like Ollama.
    // The `<|channel>`/`<channel|>` special-token convention IS still demuxed
    // below (Ollama demuxes that too).

    /// Parser state — exposed for diagnostics / tests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParseState {
        /// Default: tokens flow into the visible-content buffer.
        Visible,
        /// Inside `<|channel>thought\n…<channel|>` — reasoning buffer.
        Reasoning,
        /// Inside `<|tool_call>…<tool_call|>` — tool buffer (decoded as one
        /// blob when the closing token arrives).
        ToolCall,
    }

    /// Token-streaming parser. Lives for the lifetime of a single request.
    pub struct ResponseParser<'a> {
        template: &'a Gemma4ChatTemplate,
        state: ParseState,
        visible_tokens: Vec<u32>,
        reasoning_tokens: Vec<u32>,
        tool_tokens: Vec<u32>,
        tool_calls: Vec<ParsedToolCall>,
        /// Phase 1.6c: incremental tool-call name detection state.
        /// Cleared at each `<|tool_call>` open. Holds the decoded
        /// prefix of the current tool-call body so we can fire a
        /// `ToolCallStart{name}` event the moment `call:NAME{` is
        /// parseable — typically 10-30ms into decode vs 200-500ms
        /// at full-buffer flush.
        tool_text_prefix: String,
        /// Set true after we've fired the name event for the current
        /// `<|tool_call>…<tool_call|>` span. Prevents double-firing
        /// when more body tokens arrive after the name was already
        /// detected.
        tool_name_emitted: bool,
    }

    impl<'a> ResponseParser<'a> {
        pub fn new(template: &'a Gemma4ChatTemplate) -> Self {
            Self {
                template,
                state: ParseState::Visible,
                visible_tokens: Vec::new(),
                reasoning_tokens: Vec::new(),
                tool_tokens: Vec::new(),
                tool_calls: Vec::new(),
                tool_text_prefix: String::new(),
                tool_name_emitted: false,
            }
        }

        pub fn state(&self) -> ParseState {
            self.state
        }

        /// Feed one freshly-generated token.
        ///
        /// Returns `Ok(())`. Errors only on malformed tool-call payloads
        /// (e.g. JSON conversion failure on `<tool_call|>` flush).
        pub fn push(&mut self, token: u32) -> Result<()> {
            match (self.state, token) {
                // ── Reasoning channel ─────────────────────────────────
                (ParseState::Visible, t) if t == TOK_CHANNEL_OPEN => {
                    self.state = ParseState::Reasoning;
                }
                (ParseState::Reasoning, t) if t == TOK_CHANNEL_CLOSE => {
                    self.state = ParseState::Visible;
                }

                // ── Tool call block ───────────────────────────────────
                (ParseState::Visible, t) if t == TOK_TOOL_CALL_OPEN => {
                    self.state = ParseState::ToolCall;
                    self.tool_tokens.clear();
                    self.tool_text_prefix.clear();
                    self.tool_name_emitted = false;
                }
                (ParseState::ToolCall, t) if t == TOK_TOOL_CALL_CLOSE => {
                    let raw = self
                        .template
                        .decode(&self.tool_tokens, /* skip_special */ false)
                        .context("decode tool-call body")?;
                    for call in parse_tool_call_body(&raw)? {
                        self.tool_calls.push(call);
                    }
                    self.tool_tokens.clear();
                    self.tool_text_prefix.clear();
                    self.tool_name_emitted = false;
                    self.state = ParseState::Visible;
                }

                // ── Accumulate content tokens ─────────────────────────
                // Plain-text `<think>` is left in content (Ollama parity — see
                // the note near TOK_TOOL_CALL_*). Only `<|channel>` special
                // tokens (handled above) switch to Reasoning.
                (ParseState::Visible, _) => self.visible_tokens.push(token),
                (ParseState::Reasoning, _) => self.reasoning_tokens.push(token),
                (ParseState::ToolCall, _) => self.tool_tokens.push(token),
            }
            Ok(())
        }

        /// Phase 1.6c: incremental tool-call name detection. Append
        /// the decoded text fragment of the most recently pushed
        /// token (text decoded with `skip_special=false` so the body
        /// chars are visible) and return `Some(name)` exactly once
        /// per `<|tool_call>` span — at the moment `call:NAME{` first
        /// becomes parseable. Subsequent calls within the same span
        /// return `None`. Returns `None` at all times outside a
        /// `ToolCall` state.
        ///
        /// The HTTP / SSE layer uses this to emit the OpenAI
        /// `delta.tool_calls[].function.name` (or Anthropic
        /// `content_block_start tool_use`) chunk early, BEFORE the
        /// args body has been fully buffered. Args still emit in
        /// one chunk at `<|tool_call>` close (Approach C — args
        /// buffered, start eager).
        pub fn observe_tool_text_fragment(&mut self, fragment: &str) -> Option<String> {
            if !matches!(self.state, ParseState::ToolCall) {
                return None;
            }
            if self.tool_name_emitted {
                return None;
            }
            self.tool_text_prefix.push_str(fragment);
            // Look for the first `call:NAME{` pattern. NAME is read
            // permissively — any character until the first `{` (or
            // newline as a hard boundary) is allowed. This accepts
            // non-OpenAI-spec names that clients like Ayla may pass
            // through (e.g. MCP server prefixes containing spaces /
            // parens like `Playwright (Stealth)__browser_navigate`).
            // Trailing whitespace between NAME and `{` is trimmed.
            let prefix = self.tool_text_prefix.as_str();
            let Some(start) = prefix.find("call:") else {
                return None;
            };
            let name_start = start + "call:".len();
            let after = &prefix[name_start..];
            // Scan until '{' or newline (which acts as a hard boundary
            // so a stray `call:` token on its own line can't swallow
            // following text).
            let Some(stop_offset) = after.find(|c: char| c == '{' || c == '\n' || c == '\r') else {
                return None;
            };
            if after.as_bytes().get(stop_offset) != Some(&b'{') {
                return None;
            }
            let name = after[..stop_offset].trim();
            if name.is_empty() {
                return None;
            }
            let name = name.to_string();
            self.tool_name_emitted = true;
            Some(name)
        }

        /// Finalize the parser, decoding accumulated buffers into strings.
        pub fn finalize(self) -> Result<ParsedResponse> {
            let visible = self
                .template
                .decode(&self.visible_tokens, /* skip_special */ true)
                .context("decode visible buffer")?;
            let reasoning = self
                .template
                .decode(&self.reasoning_tokens, /* skip_special */ true)
                .context("decode reasoning buffer")?;
            Ok(ParsedResponse {
                visible,
                reasoning,
                tool_calls: self.tool_calls,
            })
        }
    }

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
            // '{' (newline is a hard boundary). Accepts non-OpenAI-spec
            // function names (spaces, parens, dots, …) so clients like
            // Ayla can pass MCP-prefixed names through without
            // sanitizing — e.g. `Playwright (Stealth)__browser_navigate`.
            let name_start = call_pos + "call:".len();
            let mut brace_start = name_start;
            while brace_start < bytes.len() {
                let b = bytes[brace_start];
                if b == b'{' || b == b'\n' || b == b'\r' {
                    break;
                }
                brace_start += 1;
            }
            if brace_start >= bytes.len() || bytes[brace_start] != b'{' {
                i = name_start;
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

        // quote bare keys. A bare key follows '{' or ',' (after
        // optional whitespace), consists of [\w-]+, and is terminated by ':'.
        let mut quoted = String::with_capacity(buf.len() + 16);
        let bytes = buf.as_bytes();
        let mut idx = 0usize;
        while idx < bytes.len() {
            let c = bytes[idx];
            quoted.push(c as char);
            if c == b'{' || c == b',' {
                // Skip whitespace.
                let mut j = idx + 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    quoted.push(bytes[j] as char);
                    j += 1;
                }
                // Identifier?
                let id_start = j;
                while j < bytes.len() {
                    let b = bytes[j];
                    if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j > id_start && j < bytes.len() && bytes[j] == b':' {
                    let ident = &buf[id_start..j];
                    quoted.push('"');
                    quoted.push_str(ident);
                    quoted.push('"');
                    idx = j;
                    continue;
                }
            }
            idx += 1;
        }

        // substitute placeholders with JSON-escaped strings.
        let mut final_str = String::with_capacity(quoted.len() + 32);
        let bytes = quoted.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == 0 {
                // \x00 N \x00 placeholder
                let mut j = i + 1;
                while j < bytes.len() && bytes[j] != 0 {
                    j += 1;
                }
                if j >= bytes.len() {
                    return Err(anyhow!("placeholder NUL mismatch"));
                }
                let n: usize = std::str::from_utf8(&bytes[i + 1..j])
                    .context("placeholder utf8")?
                    .parse()
                    .context("placeholder index")?;
                let s = strings.get(n).ok_or_else(|| anyhow!("placeholder oob"))?;
                final_str.push_str(&serde_json::to_string(s).context("escape string")?);
                i = j + 1;
            } else {
                final_str.push(bytes[i] as char);
                i += 1;
            }
        }

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

        #[test]
        fn state_machine_transitions_reasoning() {
            // We can build a parser without a real tokenizer for state-only
            // tests, but accumulator decoding needs one. Skip decode-paths
            // here; verify state transitions only.
            //
            // Note: we can't construct ResponseParser without a tokenizer,
            // so this state-only assertion lives in a dedicated test below
            // gated on the tokenizer fixture being present.
            let _ = ParseState::Visible;
        }
    }

    // ───────────────────── tokenizer-gated tests ─────────────────────

    #[cfg(test)]
    mod tokenizer_tests {
        use super::*;
        use crate::gemma4_chat::imp::Gemma4ChatTemplate;
        use std::path::Path;

        // Generic fallback locations. For local dev, point at any Gemma 4
        // `tokenizer.json` via the `LUMEN_TEST_GEMMA4_TOKENIZER` env var —
        // these `#[ignore]`d tests then run against it.
        const CANDIDATE_TOKENIZER_PATHS: &[&str] = &[
            "models/gemma-4-26b-a4b/tokenizer.json",
            "models/gemma-4-26b-a4b-mlx-4bit/tokenizer.json",
        ];

        fn load_or_skip() -> Option<Gemma4ChatTemplate> {
            let env_path = std::env::var("LUMEN_TEST_GEMMA4_TOKENIZER").ok();
            let p = env_path
                .as_deref()
                .map(Path::new)
                .filter(|p| p.exists())
                .or_else(|| {
                    CANDIDATE_TOKENIZER_PATHS
                        .iter()
                        .map(Path::new)
                        .find(|p| p.exists())
                })?;
            Some(Gemma4ChatTemplate::from_file(p).expect("load tokenizer"))
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parser_visible_only_passthrough() {
            let Some(tpl) = load_or_skip() else { return };
            // Encode "Hello world" to get real token ids.
            let ids = tpl.encode_plain("Hello world").expect("encode");
            let mut p = ResponseParser::new(&tpl);
            for id in &ids {
                p.push(*id).expect("push");
            }
            let resp = p.finalize().expect("finalize");
            assert!(
                resp.visible.contains("Hello") && resp.visible.contains("world"),
                "visible={:?}",
                resp.visible
            );
            assert!(resp.reasoning.is_empty(), "no reasoning expected");
            assert!(resp.tool_calls.is_empty());
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parser_splits_reasoning_from_visible() {
            let Some(tpl) = load_or_skip() else { return };
            // Build a stream: [<|channel> "thought\nbecause sky" <channel|>]
            //                  "Blue."
            let inside = tpl.encode_plain("thought\nbecause sky").expect("encode");
            let outside = tpl.encode_plain("Blue.").expect("encode");

            let mut p = ResponseParser::new(&tpl);
            p.push(TOK_CHANNEL_OPEN).expect("open chan");
            for id in &inside {
                p.push(*id).expect("inside");
            }
            p.push(TOK_CHANNEL_CLOSE).expect("close chan");
            for id in &outside {
                p.push(*id).expect("outside");
            }

            let resp = p.finalize().expect("finalize");
            assert!(
                resp.reasoning.contains("because"),
                "reasoning={:?}",
                resp.reasoning
            );
            assert!(resp.visible.contains("Blue"), "visible={:?}", resp.visible);
            assert!(resp.tool_calls.is_empty());
        }

        /// Ollama parity: plain-text `<think>…</think>` is NOT demuxed — it
        /// flows through verbatim as visible content (Ollama's gemma4 parser
        /// only recognizes the `<|channel>` special tokens). The `reasoning`
        /// field stays empty for this convention. This guards against
        /// re-introducing the text-tag demux that strands the reply in
        /// `reasoning` (blank bubble) when the model never closes `</think>`.
        #[test]
        #[ignore = "requires a Gemma 4 tokenizer.json (~5 MB)"]
        fn parser_keeps_text_think_in_content() {
            let Some(tpl) = load_or_skip() else { return };
            let stream = tpl
                .encode_plain("<think>\nbecause the sky scatters blue light\n</think>\nBlue.")
                .expect("encode");
            let mut p = ResponseParser::new(&tpl);
            for id in &stream {
                p.push(*id).expect("push");
            }
            let resp = p.finalize().expect("finalize");
            // Everything stays in visible content — markers included.
            assert!(
                resp.visible.contains("<think"),
                "visible={:?}",
                resp.visible
            );
            assert!(
                resp.visible.contains("scatters") && resp.visible.contains("Blue"),
                "visible={:?}",
                resp.visible
            );
            assert!(resp.reasoning.is_empty(), "reasoning={:?}", resp.reasoning);
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parser_extracts_tool_call() {
            let Some(tpl) = load_or_skip() else { return };
            // Body string (without the bracket tokens, which we'll add as
            // the special-token boundaries):
            //   call:get_weather{city:<|"|>Seoul<|"|>}
            let body = tpl
                .encode_plain(r#"call:get_weather{city:<|"|>Seoul<|"|>}"#)
                .expect("encode body");

            let mut p = ResponseParser::new(&tpl);
            p.push(TOK_TOOL_CALL_OPEN).expect("open tool");
            for id in &body {
                p.push(*id).expect("body");
            }
            p.push(TOK_TOOL_CALL_CLOSE).expect("close tool");

            let resp = p.finalize().expect("finalize");
            assert_eq!(resp.tool_calls.len(), 1, "one tool call");
            assert_eq!(resp.tool_calls[0].name, "get_weather");
            assert_eq!(resp.tool_calls[0].arguments["city"], "Seoul");
            assert!(resp.visible.is_empty(), "no visible content");
        }
    }
}
