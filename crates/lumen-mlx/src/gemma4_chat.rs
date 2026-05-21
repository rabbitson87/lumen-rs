//! Gemma 4 tokenizer + chat template.
//!
//! Wraps a HuggingFace `tokenizers::Tokenizer` (loaded from `tokenizer.json`)
//! and exposes the minimal subset of Gemma 4's chat template needed for the
//! initial OpenAI-compatible HTTP shim:
//!   • BOS / EOS / special-turn / channel / think tokens
//!   • System / user / assistant message rendering
//!   • Optional `enable_thinking` toggle (mirrors mlx_lm.server's flag)
//!   • Optional `add_generation_prompt` (set true for inference, false for
//!     loss-computation parity tests)
//!
//! Tool calls + multimodal (image/audio) blocks are NOT implemented here —
//! those land in W4 (b).
//!
//! Reference: `tokenizer_config.json`, `chat_template.jinja`, and
//! `.ai/memory/active/gemma4-26b-a4b-port/GAP.md`.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // surfaces via gemma4_chat::imp::* once the server lands
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use std::path::Path;
    use tokenizers::Tokenizer;

    // ────────── Hard-coded special-token IDs ─────────────────────────────
    // Sourced from `tokenizer.json` of `gemma-4-26b-a4b-mlx-4bit`. They are
    // model-family invariants (same across every Gemma 4 size), so encoding
    // them as constants avoids a per-request tokenizer.token_to_id() lookup.
    pub const TOK_PAD: u32 = 0;
    pub const TOK_EOS: u32 = 1;
    pub const TOK_BOS: u32 = 2;
    pub const TOK_TOOL_CALL_OPEN: u32 = 48; // <|tool_call>
    pub const TOK_TOOL_CALL_CLOSE: u32 = 49; // <tool_call|>
    pub const TOK_TOOL_RESPONSE_OPEN: u32 = 50; // <|tool_response> (also in EOS set [1,106,50])
    pub const TOK_TOOL_RESPONSE_CLOSE: u32 = 51; // <tool_response|>
    pub const TOK_QUOTE_DELIM: u32 = 52; // <|"|> — Gemma 4 custom string delim inside tool_call/tool_response args
    pub const TOK_THINK: u32 = 98;
    pub const TOK_CHANNEL_OPEN: u32 = 100; // <|channel>
    pub const TOK_CHANNEL_CLOSE: u32 = 101; // <channel|>
    pub const TOK_TURN_OPEN: u32 = 105; // <|turn>
    pub const TOK_TURN_CLOSE: u32 = 106; // <turn|>

    /// Chat role projected onto the wire-level Gemma 4 role string.
    ///
    /// `Assistant` maps to `"model"` (Google's terminology) when rendered,
    /// matching the official `chat_template.jinja`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ChatRole {
        System,
        User,
        Assistant,
    }

    impl ChatRole {
        fn wire_label(self) -> &'static str {
            match self {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
            }
        }
    }

    /// One turn in the conversation. Borrowed text avoids the per-request
    /// allocation churn that a `String` field would incur on every chat
    /// completion call.
    #[derive(Debug, Clone)]
    pub struct ChatMessage<'a> {
        pub role: ChatRole,
        pub content: &'a str,
    }

    /// Options controlling chat-template rendering.
    #[derive(Debug, Clone)]
    pub struct RenderOptions {
        /// Mirrors `enable_thinking` from `chat_template.jinja`:
        ///   • true  → inject `<|think|>` after the system header, allowing
        ///             the model to emit a `<|channel>thought…<channel|>`
        ///             block before its visible answer.
        ///   • false → pre-fill `<|channel>thought\n<channel|>` at the end
        ///             of the generation prompt so the model skips the
        ///             reasoning channel entirely.
        pub enable_thinking: bool,
        /// If true, append `<|turn>model\n…` so the next forward() call
        /// produces the assistant's reply. Set to false for loss-style
        /// parity tests where we want to score an existing assistant turn.
        pub add_generation_prompt: bool,
    }

    impl Default for RenderOptions {
        fn default() -> Self {
            Self {
                enable_thinking: false,
                add_generation_prompt: true,
            }
        }
    }

    /// Loaded tokenizer + chat-template state. Cheap to clone (it carries a
    /// single `Tokenizer` which is itself `Arc`-internally).
    pub struct Gemma4ChatTemplate {
        tokenizer: Tokenizer,
    }

    impl Gemma4ChatTemplate {
        /// Load `tokenizer.json` from a model directory (the same directory
        /// that holds `config.json` + the safetensors shards).
        pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
            let path = dir.as_ref().join("tokenizer.json");
            Self::from_file(&path)
        }

        pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
            let p = path.as_ref();
            let tokenizer =
                Tokenizer::from_file(p).map_err(|e| anyhow!("tokenizer load {p:?}: {e}"))?;
            // Sanity: confirm a couple of the constants resolve to the
            // strings we think they do. Catches a swapped tokenizer file.
            let bos = tokenizer
                .id_to_token(TOK_BOS)
                .ok_or_else(|| anyhow!("tokenizer missing id {TOK_BOS}"))?;
            if bos != "<bos>" {
                return Err(anyhow!(
                    "tokenizer id {} maps to {:?}, expected <bos>",
                    TOK_BOS,
                    bos
                ));
            }
            let turn_open = tokenizer
                .id_to_token(TOK_TURN_OPEN)
                .ok_or_else(|| anyhow!("tokenizer missing id {TOK_TURN_OPEN}"))?;
            if turn_open != "<|turn>" {
                return Err(anyhow!(
                    "tokenizer id {} maps to {:?}, expected <|turn>",
                    TOK_TURN_OPEN,
                    turn_open
                ));
            }
            Ok(Self { tokenizer })
        }

        pub fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        /// Encode raw text (no chat-template wrapping) without auto-adding
        /// BOS/EOS. Used internally by `render_to_ids` for the textual
        /// segments between special tokens.
        pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
            let enc = self
                .tokenizer
                .encode(text, /* add_special_tokens */ false)
                .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
            Ok(enc.get_ids().to_vec())
        }

        /// Decode token ids back to a string. Skips special tokens by
        /// default — useful for streaming responses where the caller wants
        /// only the visible reply.
        pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            self.tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(|e| anyhow!("tokenizer decode: {e}"))
        }

        /// Render a chat conversation into a flat token-id list.
        ///
        /// Mirrors the *textual* subset of `chat_template.jinja`:
        ///   • Single optional system message at messages[0].
        ///   • Remaining messages: user / assistant alternation.
        ///   • Thinking toggle as described on `RenderOptions::enable_thinking`.
        ///
        /// Image / audio / video blocks are rejected at this layer.
        /// Tool calls in *history* (e.g. an assistant message that previously
        /// invoked a tool) are not yet handled — Phase 1.4 owns turn-2
        /// stitching. For Phase 1.3 the model only sees the *definitions*
        /// (via `tools` argument on `render_to_ids_with_tools`) and a clean
        /// user turn.
        pub fn render_to_ids(
            &self,
            messages: &[ChatMessage<'_>],
            opts: &RenderOptions,
        ) -> Result<Vec<u32>> {
            self.render_to_ids_with_tools(messages, opts, &[])
        }

        /// Like `render_to_ids` but also injects tool *definitions* into the
        /// system turn. Empty `tools` slice produces output identical to
        /// `render_to_ids` (no system turn forced).
        ///
        /// Tools layout inside the system turn (matches the canonical
        /// `chat_template.jinja`):
        ///
        /// ```text
        /// <|turn>system
        /// [system content]
        /// <|tool>declaration:NAME{...}<tool|>
        /// <|tool>declaration:NAME2{...}<tool|>
        /// <turn|>
        /// ```
        ///
        /// When `tools` is non-empty the system turn header is forced to
        /// appear even if neither `enable_thinking` is set nor a system
        /// message is present.
        pub fn render_to_ids_with_tools(
            &self,
            messages: &[ChatMessage<'_>],
            opts: &RenderOptions,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
        ) -> Result<Vec<u32>> {
            // ── header ────────────────────────────────────────────────
            let mut out: Vec<u32> =
                Vec::with_capacity(64 + messages.len() * 32 + tools.len() * 96);
            out.push(TOK_BOS);

            let (system_msg, body) = match messages.first() {
                Some(m) if m.role == ChatRole::System => (Some(m), &messages[1..]),
                _ => (None, messages),
            };

            let has_system = system_msg.is_some();
            let has_tools = !tools.is_empty();
            let need_header = opts.enable_thinking || has_system || has_tools;
            if need_header {
                out.push(TOK_TURN_OPEN);
                out.extend(
                    self.encode_plain("system\n")
                        .context("encode 'system\\n'")?,
                );
                if opts.enable_thinking {
                    out.push(TOK_THINK);
                    out.extend(self.encode_plain("\n").context("encode '\\n'")?);
                }
                if let Some(sys) = system_msg {
                    out.extend(
                        self.encode_plain(sys.content.trim())
                            .context("encode system content")?,
                    );
                }
                if has_tools {
                    // Tool definitions follow the system content directly
                    // (the jinja just concatenates — no separator). Render
                    // to pseudo-JSON text then tokenize through the same
                    // `encode_plain` path so `<|tool>` / `<tool|>` / `<|"|>`
                    // resolve via the tokenizer's added_tokens table.
                    let tool_text = crate::gemma4_tools::imp::render_tool_definitions_text(
                        tools,
                    )
                    .context("render tool definitions text")?;
                    out.extend(
                        self.encode_plain(&tool_text)
                            .context("encode tool definitions")?,
                    );
                }
                out.push(TOK_TURN_CLOSE);
                out.extend(self.encode_plain("\n").context("encode '\\n' after sys")?);
            }

            // ── body messages ─────────────────────────────────────────
            for msg in body {
                if msg.role == ChatRole::System {
                    return Err(anyhow!(
                        "chat-template: extra system message at index > 0 not supported"
                    ));
                }
                let role = msg.role.wire_label();
                out.push(TOK_TURN_OPEN);
                out.extend(
                    self.encode_plain(&format!("{role}\n"))
                        .with_context(|| format!("encode '{role}\\n'"))?,
                );
                out.extend(
                    self.encode_plain(msg.content.trim())
                        .with_context(|| format!("encode {role} content"))?,
                );
                out.push(TOK_TURN_CLOSE);
                out.extend(self.encode_plain("\n").context("encode '\\n' after turn")?);
            }

            // ── generation prompt ─────────────────────────────────────
            if opts.add_generation_prompt {
                out.push(TOK_TURN_OPEN);
                out.extend(self.encode_plain("model\n").context("encode 'model\\n'")?);
                if !opts.enable_thinking {
                    // Pre-fill empty thought channel so the model jumps
                    // straight to the visible answer.
                    out.push(TOK_CHANNEL_OPEN);
                    out.extend(
                        self.encode_plain("thought\n")
                            .context("encode 'thought\\n'")?,
                    );
                    out.push(TOK_CHANNEL_CLOSE);
                }
            }

            Ok(out)
        }

        /// Tool-aware structured chat renderer.
        ///
        /// Accepts `ChatTurn`s (with `Assistant.tool_calls` and `Tool`
        /// variants the legacy `render_to_ids` can't represent) and emits
        /// the canonical Gemma 4 layout where an assistant's tool calls and
        /// the corresponding tool results live INSIDE a single model turn:
        ///
        /// ```text
        /// <|turn>model
        /// [reasoning]
        /// <|tool_call>call:F1{...}<tool_call|>
        /// <|tool_response>response:F1{...}<tool_response|>
        /// <|tool_call>call:F2{...}<tool_call|>
        /// <|tool_response>response:F2{...}<tool_response|>
        /// [next visible content if assistant continued]
        /// <turn|>
        /// ```
        ///
        /// The renderer walks turns linearly and groups consecutive
        /// `Tool` turns into the previous `Assistant` turn's model block.
        /// `name` for the tool response is resolved by matching
        /// `Tool.tool_call_id` against the assistant's prior
        /// `tool_calls[].id`, with `Tool.name` as a fallback when the
        /// client supplies one.
        pub fn render_chat_history(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            opts: &RenderOptions,
            tools: &[crate::chat_io::ToolDef<'_>],
        ) -> Result<Vec<u32>> {
            use crate::chat_io::ChatTurn;

            let mut out: Vec<u32> = Vec::with_capacity(64 + turns.len() * 64);
            out.push(TOK_BOS);

            // ── System / tools header ─────────────────────────────────
            let (system_text, body_start): (Option<&str>, usize) = match turns.first() {
                Some(ChatTurn::System(s)) => (Some(*s), 1),
                _ => (None, 0),
            };
            let has_tools = !tools.is_empty();
            let need_header = opts.enable_thinking || system_text.is_some() || has_tools;
            if need_header {
                out.push(TOK_TURN_OPEN);
                out.extend(
                    self.encode_plain("system\n")
                        .context("encode 'system\\n'")?,
                );
                if opts.enable_thinking {
                    out.push(TOK_THINK);
                    out.extend(self.encode_plain("\n").context("encode '\\n'")?);
                }
                if let Some(sys) = system_text {
                    out.extend(
                        self.encode_plain(sys.trim())
                            .context("encode system content")?,
                    );
                }
                if has_tools {
                    let tool_text =
                        crate::gemma4_tools::imp::render_tool_definitions_text(tools)
                            .context("render tool definitions text")?;
                    out.extend(
                        self.encode_plain(&tool_text)
                            .context("encode tool definitions")?,
                    );
                }
                out.push(TOK_TURN_CLOSE);
                out.extend(self.encode_plain("\n").context("encode '\\n' after sys")?);
            }

            // ── Body turns ─────────────────────────────────────────────
            // Walk with index; advance manually so assistant-with-tool-calls
            // can consume the trailing Tool turns into the same model turn.
            let mut i = body_start;
            while i < turns.len() {
                match &turns[i] {
                    ChatTurn::System(_) => {
                        return Err(anyhow!(
                            "render_chat_history: extra system turn at index {i} (must be first)"
                        ));
                    }
                    ChatTurn::User(text) => {
                        out.push(TOK_TURN_OPEN);
                        out.extend(
                            self.encode_plain("user\n").context("encode 'user\\n'")?,
                        );
                        out.extend(
                            self.encode_plain(text.trim())
                                .context("encode user content")?,
                        );
                        out.push(TOK_TURN_CLOSE);
                        out.extend(
                            self.encode_plain("\n").context("encode '\\n' after user")?,
                        );
                        i += 1;
                    }
                    ChatTurn::Tool { .. } => {
                        return Err(anyhow!(
                            "render_chat_history: Tool turn at index {i} has no preceding Assistant turn"
                        ));
                    }
                    ChatTurn::Assistant { text, tool_calls } => {
                        out.push(TOK_TURN_OPEN);
                        out.extend(
                            self.encode_plain("model\n")
                                .context("encode 'model\\n'")?,
                        );
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            out.extend(
                                self.encode_plain(trimmed)
                                    .context("encode assistant content")?,
                            );
                        }
                        // Emit each tool_call + the matching tool_response
                        // (if a following Tool turn answers it). Tool turns
                        // are consumed in input order; we resolve names by
                        // tool_call_id lookup but fall back to the assistant's
                        // call name when the Tool turn doesn't supply one.
                        let mut next = i + 1;
                        for tc in tool_calls.iter() {
                            out.extend(
                                self.encode_tool_call(tc.name, tc.arguments)
                                    .with_context(|| {
                                        format!("encode tool_call {:?}", tc.name)
                                    })?,
                            );
                            // Find the next Tool turn that answers this call.
                            // We accept either an exact tool_call_id match or
                            // positional fallback (consume the next Tool turn
                            // unconditionally) — some clients omit ids.
                            if let Some(ChatTurn::Tool {
                                tool_call_id,
                                name,
                                content,
                            }) = turns.get(next)
                            {
                                let _ = tool_call_id; // matched positionally
                                let resolved_name = name.unwrap_or(tc.name);
                                out.extend(
                                    self.render_tool_response_block(resolved_name, content)
                                        .with_context(|| {
                                            format!(
                                                "render tool_response for {resolved_name:?}"
                                            )
                                        })?,
                                );
                                next += 1;
                            }
                        }
                        // Consume any *trailing* unmatched Tool turns
                        // immediately after the assistant block. They
                        // typically belong to this same model turn (the
                        // client may have batched them all).
                        while let Some(ChatTurn::Tool {
                            tool_call_id: _,
                            name,
                            content,
                        }) = turns.get(next)
                        {
                            let resolved_name = name.unwrap_or("unknown");
                            out.extend(
                                self.render_tool_response_block(resolved_name, content)
                                    .with_context(|| {
                                        format!(
                                            "render trailing tool_response {resolved_name:?}"
                                        )
                                    })?,
                            );
                            next += 1;
                        }
                        out.push(TOK_TURN_CLOSE);
                        out.extend(
                            self.encode_plain("\n").context("encode '\\n' after model")?,
                        );
                        i = next;
                    }
                }
            }

            // ── Generation prompt ─────────────────────────────────────
            if opts.add_generation_prompt {
                out.push(TOK_TURN_OPEN);
                out.extend(self.encode_plain("model\n").context("encode 'model\\n'")?);
                if !opts.enable_thinking {
                    out.push(TOK_CHANNEL_OPEN);
                    out.extend(
                        self.encode_plain("thought\n")
                            .context("encode 'thought\\n'")?,
                    );
                    out.push(TOK_CHANNEL_CLOSE);
                }
            }

            Ok(out)
        }

        /// Encode one `<|tool_call>call:NAME{key:VAL,...}<tool_call|>` block.
        ///
        /// Mirrors the canonical jinja layout from
        /// `chat_template.jinja`'s assistant `message.tool_calls` branch:
        /// keys are unquoted, string values escape via `<|"|>`-pairs,
        /// numbers / booleans / arrays raw. Uses the same `format_argument`
        /// serializer the tool-definition renderer uses, with
        /// `escape_keys=false` since tool-call arguments use bare keys.
        fn encode_tool_call(
            &self,
            name: &str,
            arguments: &serde_json::Value,
        ) -> Result<Vec<u32>> {
            if name.is_empty() {
                return Err(anyhow!("encode_tool_call: empty function name"));
            }
            let mut out = Vec::with_capacity(32);
            out.push(crate::gemma4_response::imp::TOK_TOOL_CALL_OPEN);
            let body = crate::gemma4_tools::imp::format_tool_call_body(name, arguments);
            out.extend(
                self.encode_plain(&body)
                    .with_context(|| format!("encode tool_call body {name:?}"))?,
            );
            out.push(crate::gemma4_response::imp::TOK_TOOL_CALL_CLOSE);
            Ok(out)
        }

        /// Render a single Gemma 4 tool-response block.
        ///
        /// Format (mirrors `format_tool_response_block` in
        /// `chat_template.jinja`):
        ///
        /// ```text
        /// <|tool_response> response:NAME{value:<|"|>CONTENT<|"|>} <tool_response|>
        /// ```
        ///
        /// The block is intended to be appended *inside the previous model
        /// turn* (no new `<|turn>model` marker), then concatenated with the
        /// existing token sequence to drive turn-2 inference. The model
        /// continues emitting visible reply tokens immediately after the
        /// `<tool_response|>` marker.
        ///
        /// `tool_name` should match the function name the model called in
        /// the corresponding `<|tool_call>` block.
        ///
        /// **Scope note**: full tool-aware chat-template rendering (tool
        /// *definitions* in the system header, multiple tool_calls per
        /// assistant turn, multimodal blocks) is deferred to Phase 2's
        /// TaskWeaver integration. This helper is the minimum primitive
        /// the W6 (v) gate requires — it lets a caller stitch together
        /// turn-1 output + tool result + turn-2 continuation manually.
        pub fn render_tool_response_block(
            &self,
            tool_name: &str,
            response_text: &str,
        ) -> Result<Vec<u32>> {
            if tool_name.is_empty() {
                return Err(anyhow!("render_tool_response_block: empty tool_name"));
            }
            let mut out: Vec<u32> = Vec::with_capacity(16 + response_text.len());
            out.push(TOK_TOOL_RESPONSE_OPEN);
            out.extend(
                self.encode_plain(&format!("response:{tool_name}{{value:"))
                    .with_context(|| format!("encode tool_response header for {tool_name:?}"))?,
            );
            out.push(TOK_QUOTE_DELIM);
            out.extend(
                self.encode_plain(response_text)
                    .context("encode tool_response content")?,
            );
            out.push(TOK_QUOTE_DELIM);
            out.extend(
                self.encode_plain("}")
                    .context("encode tool_response close brace")?,
            );
            out.push(TOK_TOOL_RESPONSE_CLOSE);
            Ok(out)
        }
    }

    // ───────────────────────── tests ─────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::path::Path;

        const LMSTUDIO_TOKENIZER_PATH: &str =
            "/path/to/models/gemma-4-26b-a4b-mlx-4bit/tokenizer.json";

        fn load_template_if_present() -> Option<Gemma4ChatTemplate> {
            let p = Path::new(LMSTUDIO_TOKENIZER_PATH);
            if !p.exists() {
                eprintln!("skip: tokenizer not present at {LMSTUDIO_TOKENIZER_PATH}");
                return None;
            }
            Some(Gemma4ChatTemplate::from_file(p).expect("load tokenizer"))
        }

        #[test]
        fn constants_match_documented_ids() {
            // Compile-time sanity — these are documented in
            // GAP.md and tokenizer.json. If the constants drift, the
            // chat template will produce subtly broken sequences.
            assert_eq!(TOK_PAD, 0);
            assert_eq!(TOK_EOS, 1);
            assert_eq!(TOK_BOS, 2);
            assert_eq!(TOK_TOOL_CALL_OPEN, 48);
            assert_eq!(TOK_TOOL_CALL_CLOSE, 49);
            assert_eq!(TOK_TOOL_RESPONSE_OPEN, 50);
            assert_eq!(TOK_TOOL_RESPONSE_CLOSE, 51);
            assert_eq!(TOK_QUOTE_DELIM, 52);
            assert_eq!(TOK_THINK, 98);
            assert_eq!(TOK_CHANNEL_OPEN, 100);
            assert_eq!(TOK_CHANNEL_CLOSE, 101);
            assert_eq!(TOK_TURN_OPEN, 105);
            assert_eq!(TOK_TURN_CLOSE, 106);
        }

        /// tool_response block token structure.
        ///
        /// Asserts that `render_tool_response_block` produces a sequence
        /// matching the Gemma 4 chat-template format used by mlx-lm:
        ///   `<|tool_response> response:NAME{value:<|"|>CONTENT<|"|>} <tool_response|>`
        ///
        /// This is the building block for turn-2 tool roundtrip: the
        /// caller appends this to the turn-1 model output (which ended
        /// at `<|tool_response>=50` EOS), then continues generation. Full
        /// chat-template tool-definition rendering (system header) is
        /// deferred to Phase 2 TaskWeaver integration.
        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn render_tool_response_block_has_expected_structure() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let ids = tpl
                .render_tool_response_block("get_weather", "20C sunny")
                .expect("render tool_response");

            // Bookend tokens
            assert_eq!(
                ids[0], TOK_TOOL_RESPONSE_OPEN,
                "block must open with <|tool_response>=50"
            );
            assert_eq!(
                *ids.last().unwrap(),
                TOK_TOOL_RESPONSE_CLOSE,
                "block must close with <tool_response|>=51"
            );

            // Exactly two TOK_QUOTE_DELIM tokens wrap the value.
            let n_quote = ids.iter().filter(|&&t| t == TOK_QUOTE_DELIM).count();
            assert_eq!(
                n_quote, 2,
                "expected exactly two <|\"|> delim tokens around value, got {n_quote}"
            );

            // Decode (with specials) and check the textual body matches the
            // Gemma 4 chat-template format. We use skip_special=false so
            // the delim tokens are visible.
            let decoded = tpl.decode(&ids, /* skip_special */ false).expect("decode");
            assert!(
                decoded.contains("response:get_weather"),
                "decoded body must contain 'response:get_weather'; got {decoded:?}"
            );
            assert!(
                decoded.contains("20C sunny"),
                "decoded body must contain the response payload; got {decoded:?}"
            );
            assert!(
                decoded.contains("{value:") && decoded.contains("}"),
                "decoded body must wrap value in braces; got {decoded:?}"
            );
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn render_tool_response_block_rejects_empty_name() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let err = tpl
                .render_tool_response_block("", "anything")
                .expect_err("empty name must error");
            assert!(
                err.to_string().contains("empty tool_name"),
                "unexpected error: {err}"
            );
        }

        /// turn-2 stitching: turn-1 prompt + simulated
        /// model tool_call output + tool response + check turn-2 input
        /// shape (tokenized correctly, ends at expected boundary).
        ///
        /// This is a *structural* test of the turn-2 concatenation pattern
        /// the Phase 2 TaskWeaver integration will use:
        ///
        ///   1. Render turn-1 prompt with `render_to_ids`.
        ///   2. Append the simulated assistant tool_call output (synthesized
        ///      here so the test runs without the model — Phase 2 will get
        ///      real ones from `Gemma4Backend::chat`).
        ///   3. Append the tool_response block via `render_tool_response_block`.
        ///   4. Verify the resulting sequence: starts with BOS, contains the
        ///      `<|tool_call>...<tool_call|>` and `<|tool_response>...<tool_response|>`
        ///      markers in the correct order, and is suitable for feeding
        ///      back to the model as a continuation input.
        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn turn_2_stitching_assistant_toolcall_plus_response() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };

            // Turn 1: user asks for weather (no tool definitions rendered
            // here — those land with Phase 2).
            let msgs = [ChatMessage {
                role: ChatRole::User,
                content: "What's the weather in Seoul?",
            }];
            let mut convo = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect("turn 1 render");
            let turn1_len = convo.len();

            // Simulated model output: tool_call block (Phase 2 will get
            // this from real generation; here we synthesize the canonical
            // token sequence so the test is hermetic).
            convo.push(TOK_TOOL_CALL_OPEN);
            convo.extend(
                tpl.encode_plain("call:get_weather{city:")
                    .expect("encode call body"),
            );
            convo.push(TOK_QUOTE_DELIM);
            convo.extend(tpl.encode_plain("Seoul").expect("encode arg"));
            convo.push(TOK_QUOTE_DELIM);
            convo.extend(tpl.encode_plain("}").expect("encode close"));
            convo.push(TOK_TOOL_CALL_CLOSE);

            // Tool response (Phase 2's job to supply real content from the
            // executed tool; W6 (v) just validates the wire format).
            let resp = tpl
                .render_tool_response_block("get_weather", "20C sunny")
                .expect("render tool_response");
            let resp_start = convo.len();
            convo.extend(resp);

            // Invariants
            assert_eq!(convo[0], TOK_BOS, "starts with <bos>");
            assert!(convo.len() > turn1_len, "turn-2 input strictly longer");

            // Order: tool_call_open must precede tool_response_open
            let pos_call_open = convo
                .iter()
                .position(|&t| t == TOK_TOOL_CALL_OPEN)
                .expect("contains tool_call_open");
            let pos_resp_open = convo
                .iter()
                .position(|&t| t == TOK_TOOL_RESPONSE_OPEN)
                .expect("contains tool_response_open");
            assert!(
                pos_call_open < pos_resp_open,
                "tool_call ({pos_call_open}) must precede tool_response ({pos_resp_open})"
            );

            // Resp block boundary
            assert_eq!(
                convo[resp_start], TOK_TOOL_RESPONSE_OPEN,
                "appended block starts at <|tool_response>"
            );
            assert_eq!(
                *convo.last().unwrap(),
                TOK_TOOL_RESPONSE_CLOSE,
                "turn-2 input ends at <tool_response|>"
            );
        }

        #[test]
        fn render_options_default_matches_inference_path() {
            let o = RenderOptions::default();
            assert!(!o.enable_thinking);
            assert!(o.add_generation_prompt);
        }

        #[test]
        fn chat_role_wire_labels() {
            assert_eq!(ChatRole::System.wire_label(), "system");
            assert_eq!(ChatRole::User.wire_label(), "user");
            // The official Gemma 4 chat_template.jinja maps assistant→model.
            assert_eq!(ChatRole::Assistant.wire_label(), "model");
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn renders_user_only_with_generation_prompt() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [ChatMessage {
                role: ChatRole::User,
                content: "Hello",
            }];
            let ids = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect("render");
            // Expected backbone (special-token only):
            //   BOS, <|turn>, "user\n", "Hello", <turn|>, "\n",
            //   <|turn>, "model\n", <|channel>, "thought\n", <channel|>
            assert_eq!(ids[0], TOK_BOS, "starts with <bos>");
            assert!(
                ids.contains(&TOK_TURN_OPEN),
                "contains at least one <|turn> opener"
            );
            assert!(
                ids.contains(&TOK_TURN_CLOSE),
                "contains at least one <turn|> closer"
            );
            assert_eq!(
                *ids.last().unwrap(),
                TOK_CHANNEL_CLOSE,
                "non-thinking gen prompt ends with <channel|>"
            );
            // No <|think|> when thinking disabled.
            assert!(
                !ids.contains(&TOK_THINK),
                "<|think|> must not appear when enable_thinking=false"
            );
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn renders_system_and_user_with_thinking_enabled() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [
                ChatMessage {
                    role: ChatRole::System,
                    content: "Be concise.",
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi",
                },
            ];
            let ids = tpl
                .render_to_ids(
                    &msgs,
                    &RenderOptions {
                        enable_thinking: true,
                        add_generation_prompt: true,
                    },
                )
                .expect("render");
            assert_eq!(ids[0], TOK_BOS);
            assert!(
                ids.contains(&TOK_THINK),
                "<|think|> must appear when enable_thinking=true"
            );
            // With thinking enabled the gen prompt does NOT pre-fill the
            // channel — last token should be the role string's tail, not
            // <channel|>.
            assert_ne!(*ids.last().unwrap(), TOK_CHANNEL_CLOSE);
            assert!(ids.contains(&TOK_TURN_OPEN), "contains <|turn> openers");
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn renders_without_generation_prompt() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [
                ChatMessage {
                    role: ChatRole::User,
                    content: "Q",
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "A",
                },
            ];
            let ids = tpl
                .render_to_ids(
                    &msgs,
                    &RenderOptions {
                        enable_thinking: false,
                        add_generation_prompt: false,
                    },
                )
                .expect("render");
            // Should end with the assistant's closing <turn|> + newline,
            // not with any model-prompt scaffolding.
            assert!(ids.contains(&TOK_TURN_CLOSE));
            assert!(
                !ids.contains(&TOK_CHANNEL_OPEN),
                "no <|channel> without gen prompt"
            );
            assert!(
                !ids.contains(&TOK_CHANNEL_CLOSE),
                "no <channel|> without gen prompt"
            );
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn decode_round_trip_skips_specials() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let plain = "Hello, world!";
            let ids = tpl.encode_plain(plain).expect("encode plain");
            let decoded = tpl
                .decode(&ids, /* skip_special_tokens */ true)
                .expect("decode");
            // Allow whitespace shrinkage but require the content survives.
            assert!(decoded.contains("Hello"), "decoded={decoded:?}");
            assert!(decoded.contains("world"), "decoded={decoded:?}");
        }

        /// Golden token-id parity against
        /// `transformers.AutoTokenizer.apply_chat_template`.
        ///
        /// Captured 2026-05-12 against
        /// `/path/to/models/gemma-4-26b-a4b-mlx-4bit/tokenizer.json`
        /// using the official `chat_template.jinja`. These vectors are the
        /// ground truth — if we ever diverge from them, mlx_lm.server-style
        /// behavior will drift silently.
        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parity_user_only_no_thinking() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [ChatMessage {
                role: ChatRole::User,
                content: "Hello",
            }];
            let ids = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect("render");
            // HuggingFace golden:
            //   apply_chat_template([user='Hello'], add_generation_prompt=True,
            //                       enable_thinking=False)
            let golden: Vec<u32> = vec![
                2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
            ];
            assert_eq!(ids, golden, "user-only no-think parity");
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parity_system_user_thinking_enabled() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [
                ChatMessage {
                    role: ChatRole::System,
                    content: "Be concise.",
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi",
                },
            ];
            let ids = tpl
                .render_to_ids(
                    &msgs,
                    &RenderOptions {
                        enable_thinking: true,
                        add_generation_prompt: true,
                    },
                )
                .expect("render");
            let golden: Vec<u32> = vec![
                2, 105, 9731, 107, 98, 107, 3912, 63510, 236761, 106, 107, 105, 2364, 107, 10979,
                106, 107, 105, 4368, 107,
            ];
            assert_eq!(ids, golden, "sys+user think-enabled parity");
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn parity_user_assistant_no_generation_prompt() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [
                ChatMessage {
                    role: ChatRole::User,
                    content: "Q",
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "A",
                },
            ];
            let ids = tpl
                .render_to_ids(
                    &msgs,
                    &RenderOptions {
                        enable_thinking: false,
                        add_generation_prompt: false,
                    },
                )
                .expect("render");
            let golden: Vec<u32> = vec![
                2, 105, 2364, 107, 236935, 106, 107, 105, 4368, 107, 236776, 106, 107,
            ];
            assert_eq!(ids, golden, "u+a no-gen-prompt parity");
        }

        #[test]
        #[ignore = "requires tokenizer.json from lmstudio shards (~5 MB)"]
        fn rejects_extra_system_message_mid_conversation() {
            let Some(tpl) = load_template_if_present() else {
                return;
            };
            let msgs = [
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi",
                },
                ChatMessage {
                    role: ChatRole::System,
                    content: "Mid-conversation system not allowed",
                },
            ];
            let err = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect_err("must reject");
            let s = format!("{err:?}");
            assert!(
                s.contains("extra system message"),
                "expected rejection, got {s}"
            );
        }
    }
}
