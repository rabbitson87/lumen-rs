//! Generic Jinja2-based chat template renderer.
//!
//! Loads the model's upstream `chat_template.jinja` (the same file Hugging
//! Face's `transformers.AutoTokenizer.apply_chat_template` evaluates) and
//! renders messages through a `minijinja` interpreter, eliminating the
//! maintenance burden of mirroring jinja semantics in Rust.
//!
//! Designed to slot in behind the family-specific backends (Gemma 4 first,
//! Qwen 3.5/3.6 next) so each backend can either continue using its
//! hand-port (fast path) or delegate to this module (single source of truth
//! against upstream).
//!
//! The output is a `Vec<u32>` token-id sequence — the same shape
//! `Gemma4ChatTemplate::render_to_ids` returns — so callers can swap
//! renderers without touching downstream sampling / cache code.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)]
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use minijinja::{Environment, value::Value as MJValue};
    use serde_json::{Map, Value as JsonValue, json};
    use std::path::Path;
    use tokenizers::Tokenizer;

    use crate::chat_io::{ChatTurn, ToolDef};

    /// Options controlling chat-template rendering. Mirrors the subset of
    /// jinja globals consumed by Gemma 4 / Qwen templates.
    #[derive(Debug, Clone, Default)]
    pub struct JinjaRenderOptions {
        /// Maps to the `enable_thinking` jinja global. When the template
        /// branches on it (Gemma 4 does, Qwen 3.5 does, Qwen 3.6 does),
        /// the flag controls whether the model is invited to emit a
        /// reasoning channel before its visible answer.
        pub enable_thinking: bool,
        /// Maps to the `add_generation_prompt` jinja global. Set true for
        /// inference, false when scoring an existing assistant turn.
        pub add_generation_prompt: bool,
    }

    /// Loaded jinja chat template + tokenizer pair.
    ///
    /// Holds a long-lived `minijinja::Environment` that pre-parses the
    /// template once at construction; each `render_to_ids` call only pays
    /// the cost of evaluating the AST.
    pub struct JinjaChatTemplate {
        env: Environment<'static>,
        tokenizer: Tokenizer,
        bos_token: String,
        /// Cached owned template source (Environment::add_template
        /// borrows it for the lifetime of the env).
        _template_src: String,
    }

    impl JinjaChatTemplate {
        /// Load from a model directory that contains both
        /// `chat_template.jinja` and `tokenizer.json` (+ `tokenizer_config.json`
        /// for the `bos_token` literal).
        pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
            let dir = dir.as_ref();
            let template_path = dir.join("chat_template.jinja");
            let tokenizer_path = dir.join("tokenizer.json");
            let config_path = dir.join("tokenizer_config.json");

            let template_src = std::fs::read_to_string(&template_path)
                .with_context(|| format!("read chat_template.jinja at {template_path:?}"))?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("tokenizer load {tokenizer_path:?}: {e}"))?;

            let bos_token = read_bos_token(&config_path)
                .with_context(|| format!("read bos_token from {config_path:?}"))?;

            // Heap-allocate the template source and leak it for the
            // 'static lifetime that minijinja's add_template requires.
            // Cheaper than wrapping the whole Environment in a self-
            // referential struct, and the template is loaded once per
            // backend lifetime.
            let leaked: &'static str = Box::leak(template_src.clone().into_boxed_str());
            let mut env = Environment::new();
            // Trim & lstrip control match jinja2 defaults used by HF's
            // apply_chat_template: trim_blocks=True, lstrip_blocks=True
            // would over-strip; the actual transformers behavior is
            // closer to the minijinja default (no trim, no lstrip). The
            // chat_template.jinja explicitly uses `{%-` / `-%}` to
            // request whitespace control where needed.
            //
            // Register the pycompat unknown-method fallback so the
            // upstream template can call `.get()`, `.split()`,
            // `.startswith()` etc. — Python idioms that real Jinja2
            // inherits from its host language but minijinja doesn't
            // expose by default.
            env.set_unknown_method_callback(
                minijinja_contrib::pycompat::unknown_method_callback,
            );
            env.add_template("chat", leaked)
                .map_err(|e| anyhow!("parse chat_template.jinja: {e}"))?;

            Ok(Self {
                env,
                tokenizer,
                bos_token,
                _template_src: template_src,
            })
        }

        pub fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        /// Render conversation → final UTF-8 string (debug / inspection).
        /// Production callers should prefer `render_to_ids` so they don't
        /// pay the encode cost twice.
        pub fn render_to_string(
            &self,
            messages: &[ChatTurn<'_>],
            opts: &JinjaRenderOptions,
            tools: Option<&[ToolDef<'_>]>,
        ) -> Result<String> {
            let context = self.build_context(messages, opts, tools)?;
            let tmpl = self
                .env
                .get_template("chat")
                .map_err(|e| anyhow!("get template: {e}"))?;
            tmpl.render(context)
                .map_err(|e| anyhow!("render: {e}"))
        }

        /// Render conversation → flat token-id list. Equivalent in
        /// semantics to `Gemma4ChatTemplate::render_to_ids_with_tools`
        /// when the loaded template is Gemma 4's.
        pub fn render_to_ids(
            &self,
            messages: &[ChatTurn<'_>],
            opts: &JinjaRenderOptions,
            tools: Option<&[ToolDef<'_>]>,
        ) -> Result<Vec<u32>> {
            let s = self.render_to_string(messages, opts, tools)?;
            let enc = self
                .tokenizer
                .encode(s, /* add_special_tokens */ false)
                .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
            Ok(enc.get_ids().to_vec())
        }

        fn build_context(
            &self,
            messages: &[ChatTurn<'_>],
            opts: &JinjaRenderOptions,
            tools: Option<&[ToolDef<'_>]>,
        ) -> Result<MJValue> {
            let messages_json = turns_to_json(messages)?;
            let tools_json = match tools {
                Some(ts) if !ts.is_empty() => tools_to_json(ts)?,
                _ => JsonValue::Null,
            };

            let ctx = json!({
                "bos_token": self.bos_token,
                "messages": messages_json,
                "tools": tools_json,
                "enable_thinking": opts.enable_thinking,
                "add_generation_prompt": opts.add_generation_prompt,
            });
            Ok(MJValue::from_serialize(&ctx))
        }
    }

    fn read_bos_token(config_path: &Path) -> Result<String> {
        let raw = std::fs::read_to_string(config_path)?;
        let v: JsonValue = serde_json::from_str(&raw)?;
        match v.get("bos_token") {
            Some(JsonValue::String(s)) => Ok(s.clone()),
            // Some tokenizer_config.json shapes embed bos_token as an
            // object with a "content" field — Llama-style. Fall through.
            Some(JsonValue::Object(obj)) => obj
                .get("content")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow!("bos_token object missing 'content' field")),
            Some(other) => Err(anyhow!("bos_token has unexpected type: {other:?}")),
            None => Err(anyhow!("tokenizer_config.json missing bos_token field")),
        }
    }

    fn turns_to_json(turns: &[ChatTurn<'_>]) -> Result<JsonValue> {
        let mut out = Vec::with_capacity(turns.len() + turns.iter().filter(|t| matches!(t, ChatTurn::Assistant { tool_calls, .. } if !tool_calls.is_empty())).count());
        for t in turns {
            match t {
                ChatTurn::System(text) => {
                    out.push(json!({
                        "role": "system",
                        "content": *text,
                    }));
                }
                ChatTurn::User(text) => {
                    out.push(json!({
                        "role": "user",
                        "content": *text,
                    }));
                }
                ChatTurn::Assistant { text, tool_calls } => {
                    let mut msg = Map::new();
                    msg.insert("role".into(), JsonValue::String("assistant".into()));
                    msg.insert("content".into(), JsonValue::String((*text).to_string()));
                    if !tool_calls.is_empty() {
                        let tc_arr: Vec<JsonValue> = tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }
                                })
                            })
                            .collect();
                        msg.insert("tool_calls".into(), JsonValue::Array(tc_arr));
                    }
                    out.push(JsonValue::Object(msg));
                }
                ChatTurn::Tool {
                    tool_call_id,
                    name,
                    content,
                } => {
                    let mut msg = Map::new();
                    msg.insert("role".into(), JsonValue::String("tool".into()));
                    msg.insert(
                        "tool_call_id".into(),
                        JsonValue::String((*tool_call_id).to_string()),
                    );
                    if let Some(n) = name {
                        msg.insert("name".into(), JsonValue::String((*n).to_string()));
                    }
                    msg.insert("content".into(), JsonValue::String((*content).to_string()));
                    out.push(JsonValue::Object(msg));
                }
            }
        }
        Ok(JsonValue::Array(out))
    }

    fn tools_to_json(tools: &[ToolDef<'_>]) -> Result<JsonValue> {
        let arr: Vec<JsonValue> = tools
            .iter()
            .map(|t| {
                let mut fn_obj = Map::new();
                fn_obj.insert("name".into(), JsonValue::String(t.name.to_string()));
                if let Some(d) = t.description {
                    fn_obj.insert("description".into(), JsonValue::String(d.to_string()));
                }
                if let Some(p) = t.parameters {
                    fn_obj.insert("parameters".into(), p.clone());
                }
                if let Some(r) = t.response {
                    fn_obj.insert("response".into(), r.clone());
                }
                json!({
                    "type": "function",
                    "function": JsonValue::Object(fn_obj),
                })
            })
            .collect();
        Ok(JsonValue::Array(arr))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Locate the Gemma 4 4-bit IT model directory. We prefer the
        /// flat mlx-community layout (which ships chat_template.jinja
        /// alongside tokenizer.json + tokenizer_config.json).
        fn gemma4_4bit_dir() -> Option<std::path::PathBuf> {
            let home = std::env::var("HOME").ok()?;
            let p = std::path::PathBuf::from(&home)
                .join("models")
                .join("mlx-community--gemma-4-26b-a4b-it-4bit");
            if p.join("chat_template.jinja").exists()
                && p.join("tokenizer.json").exists()
                && p.join("tokenizer_config.json").exists()
            {
                Some(p)
            } else {
                None
            }
        }

        // Parity vs hand-port's golden fixture from gemma4_parity.json:
        //   messages = [{role:"user", content:"Hi"}]
        //   thinking = false, add_generation_prompt = true
        //   expected = [2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_user_only_no_thinking_matches_fixture() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => {
                    eprintln!("skipping: Gemma 4 4-bit model dir not found");
                    return;
                }
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");

            let turns = vec![ChatTurn::User("Hi")];
            let opts = JinjaRenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            };
            let s = tmpl
                .render_to_string(&turns, &opts, None)
                .expect("render to string");
            eprintln!("rendered string: {s:?}");

            let ids = tmpl
                .render_to_ids(&turns, &opts, None)
                .expect("render to ids");
            let expected: Vec<u32> = vec![
                2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
            ];
            assert_eq!(ids, expected, "rendered ids must match HF apply_chat_template");
        }

        /// Golden parity: User "Hello" + thinking=false + add_gen=true.
        /// Source: gemma4_chat.rs::parity_user_only_no_thinking — captured
        /// against HF AutoTokenizer.apply_chat_template.
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_parity_user_only_hello() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => return,
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");
            let turns = vec![ChatTurn::User("Hello")];
            let ids = tmpl
                .render_to_ids(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: false,
                        add_generation_prompt: true,
                    },
                    None,
                )
                .expect("render");
            let golden: Vec<u32> = vec![
                2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
            ];
            assert_eq!(ids, golden);
        }

        /// Golden parity: System "Be concise." + User "Hi" + thinking=true
        /// + add_gen=true. Source: gemma4_chat.rs::parity_system_user_thinking_enabled.
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_parity_system_user_thinking() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => return,
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");
            let turns = vec![
                ChatTurn::System("Be concise."),
                ChatTurn::User("Hi"),
            ];
            let ids = tmpl
                .render_to_ids(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: true,
                        add_generation_prompt: true,
                    },
                    None,
                )
                .expect("render");
            let golden: Vec<u32> = vec![
                2, 105, 9731, 107, 98, 107, 3912, 63510, 236761, 106, 107, 105, 2364, 107, 10979,
                106, 107, 105, 4368, 107,
            ];
            assert_eq!(ids, golden);
        }

        /// Golden parity: User "Q" + Assistant "A" + thinking=false +
        /// add_gen=false. Source: gemma4_chat.rs::parity_user_assistant_no_generation_prompt.
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_parity_user_assistant_no_gen_prompt() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => return,
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");
            let turns = vec![
                ChatTurn::User("Q"),
                ChatTurn::Assistant {
                    text: "A",
                    tool_calls: &[],
                },
            ];
            let ids = tmpl
                .render_to_ids(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: false,
                        add_generation_prompt: false,
                    },
                    None,
                )
                .expect("render");
            let golden: Vec<u32> = vec![
                2, 105, 2364, 107, 236935, 106, 107, 105, 4368, 107, 236776, 106, 107,
            ];
            assert_eq!(ids, golden);
        }

        /// Tools injection: a single weather-fetch tool definition lands as
        /// `<|tool>declaration:get_weather{...}<tool|>` inside the system
        /// turn. We cross-check the rendered string contains the canonical
        /// declaration shape — there's no HF golden vector for this case in
        /// the existing fixture, so the assertion is structural rather than
        /// id-by-id.
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_tools_injection_emits_declaration_block() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => return,
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");
            let params: serde_json::Value = serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to look up.",
                    }
                },
                "required": ["city"],
            });
            let tool = ToolDef {
                name: "get_weather",
                description: Some("Get the current weather for a city."),
                parameters: Some(&params),
                response: None,
            };
            let turns = vec![ChatTurn::User("Weather in Tokyo?")];
            let s = tmpl
                .render_to_string(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: false,
                        add_generation_prompt: true,
                    },
                    Some(&[tool]),
                )
                .expect("render");

            assert!(s.starts_with("<bos><|turn>system\n"));
            assert!(s.contains("<|tool>declaration:get_weather"));
            assert!(s.contains("<|\"|>The current weather for a city.<|\"|>")
                || s.contains("<|\"|>Get the current weather for a city.<|\"|>"));
            assert!(s.contains("<|turn>user\nWeather in Tokyo?<turn|>"));
            assert!(s.trim_end().ends_with("<channel|>"));
        }

        // Parity vs golden fixture: system + user + thinking=true
        //   prompt = [2, 105, 9731, 107, 98, 107, ..., 106, 107, 105, 4368, 107]
        // (See gemma4_chat.rs::parity_system_user_thinking_enabled at L771.)
        #[test]
        #[ignore = "requires local Gemma 4 model dir; run with --ignored"]
        fn gemma4_system_user_thinking_renders_think_marker() {
            let dir = match gemma4_4bit_dir() {
                Some(d) => d,
                None => {
                    eprintln!("skipping: Gemma 4 4-bit model dir not found");
                    return;
                }
            };
            let tmpl = JinjaChatTemplate::from_dir(&dir).expect("load template");

            let turns = vec![
                ChatTurn::System("You are helpful."),
                ChatTurn::User("Hi"),
            ];
            let opts = JinjaRenderOptions {
                enable_thinking: true,
                add_generation_prompt: true,
            };
            let s = tmpl
                .render_to_string(&turns, &opts, None)
                .expect("render to string");

            // Sanity: the rendered prompt must contain the literal
            // think marker AND a system turn AND a final model turn.
            assert!(
                s.starts_with("<bos><|turn>system\n<|think|>\n"),
                "system + thinking header expected at start; got: {s:?}"
            );
            assert!(s.contains("You are helpful."), "system content missing");
            assert!(s.contains("<|turn>user\nHi<turn|>"), "user turn missing");
            // With enable_thinking=true the generation prompt is just
            // <|turn>model\n (no pre-filled <|channel>thought\n<channel|>).
            assert!(
                s.trim_end().ends_with("<|turn>model"),
                "model gen prompt expected at end with thinking on; got tail: {:?}",
                &s[s.len().saturating_sub(40)..]
            );

            let ids = tmpl
                .render_to_ids(&turns, &opts, None)
                .expect("render to ids");
            // First 6 ids must be: <bos> <|turn> "system" \n <|think|> \n
            assert_eq!(&ids[..6], &[2, 105, 9731, 107, 98, 107]);
            // Last id must be the linebreak after <|turn>model.
            assert_eq!(*ids.last().unwrap(), 107);
        }
    }
}
