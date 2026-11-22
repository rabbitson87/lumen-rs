//! Serving the Inkling 2 text model: a prompt from chat turns, a generation loop, and the caches
//! of the last conversation kept for the next request.
//!
//! One conversation is retained between requests. A request whose prompt starts with tokens the
//! retained caches have already seen prefills only the rest. The caches can go back to a point
//! only if the convolution states of that point were recorded (the KV caches only append, so they
//! can be cut to any length, the states cannot), so the backend records them at the three places
//! a next prompt is likely to start from: one token before the end of the prompt (so a repeated
//! prompt can still be fed its last token and give logits), the end of the prompt, and the end of
//! the turn the model wrote (so a client that sends the reply back, reasoning included, starts
//! from the end of it).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use lumen_core::sampling::{SamplingConfig, Xorshift64, sample_from_logits};
use mlx_rs::Array;

use super::chat::{Effort, render_prompt};
use super::model::{ConvSnapshot, Inkling2Cache, Inkling2Model, LoadOptions};
use super::response::{Markers, ResponseParser};
use crate::SamplingOverrides;
use crate::chat_io::{
    BackendStreamEvent, ChatTurn, ParsedResponse, ResolvedToolChoice, StopCause, ToolDef,
};
use crate::text_tokenizer::{TextTokenizer, TokenizeStats};

/// Tokens a prefill feeds at a time. The activations of one chunk are bounded by its length, the
/// whole prompt's would not be.
///
/// Measured on a 5,167-token prompt (M3 Max), allocator peak and prefill time at chunks of
/// 128 / 256 / 512 / 1024 / 2048: 16.0 / 16.5 / 17.2 / 18.0 / 19.5 GiB and 15.3 / 16.5 / 12.8 /
/// 13.2 / 18.4 s, on 15.5 GiB of weights. Unlike the Qwen path, where a smaller chunk costs
/// latency, here the unfused masked attention over a long chunk is the expensive part, so 512 is
/// both the fastest and nearly the lowest.
const PREFILL_CHUNK: usize = 512;

/// The longest prompt plus reply served. Every one of the 48 layer visits keeps a KV cache, so a
/// token costs 192 KiB (the 8 visits of the global layers 5, 11, 17, 23 hold all tokens, the 40
/// local ones only the window, but this backend keeps them all): 64K tokens are 12 GiB.
const MAX_CONTEXT_CAP: usize = 65_536;

/// What the last request cost, for the log and for measuring what the retained caches save.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStats {
    pub prompt_tokens: usize,
    /// Prompt tokens taken from the retained caches instead of being fed.
    pub reused_tokens: usize,
    pub prefill_ms: f64,
    pub generated_tokens: usize,
    pub decode_ms: f64,
}

/// A conversation's caches, to be picked up by the next request.
struct Retained {
    /// What the caches have seen.
    tokens: Vec<u32>,
    cache: Inkling2Cache,
    /// Lengths the caches can go back to, ascending, with their convolution states.
    points: Vec<(usize, ConvSnapshot)>,
}

pub struct Inkling2Backend {
    model: Inkling2Model,
    tokenizer: Arc<TextTokenizer>,
    markers: Markers,
    eos: Vec<u32>,
    model_id: String,
    max_context: usize,
    retained: Option<Retained>,
    last_run: RunStats,
}

/// The longest recorded point the caches can go back to for a prompt `new`: its tokens are a
/// prefix of `new` and shorter than it (at least one token is left to feed, since the logits of
/// the next token come from feeding).
fn reusable_point(
    lengths: impl Iterator<Item = usize>,
    seen: &[u32],
    new: &[u32],
) -> Option<usize> {
    lengths
        .filter(|&len| len > 0 && len < new.len() && len <= seen.len())
        .filter(|&len| seen[..len] == new[..len])
        .max()
}

impl Inkling2Backend {
    /// Loads the checkpoint in `dir`: the weights, the tokenizer and its markers.
    pub fn from_dir(model_id: &str, dir: &Path) -> Result<Self> {
        let model = Inkling2Model::load(dir, LoadOptions::default())
            .with_context(|| format!("load the Inkling 2 text model from {}", dir.display()))?;
        let tokenizer = Arc::new(
            TextTokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| anyhow!("tokenizer.json: {e}"))?,
        );
        let markers = Markers::lookup(|s| tokenizer.single_token_id(s))?;
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let declared_eos = config["eos_token_id"].as_u64().map(|v| v as u32);
        ensure!(
            declared_eos == Some(markers.end_sampling),
            "config.json eos_token_id {declared_eos:?} is not the end-of-turn marker {}",
            markers.end_sampling
        );
        let model_max = config["text_config"]["model_max_length"]
            .as_u64()
            .map_or(MAX_CONTEXT_CAP, |v| v as usize);
        Ok(Self {
            model,
            tokenizer,
            markers,
            eos: vec![markers.end_sampling],
            model_id: model_id.to_string(),
            max_context: model_max.min(MAX_CONTEXT_CAP),
            retained: None,
            last_run: RunStats::default(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The longest prompt (plus reply) this backend serves: the KV cache of every token is kept.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos
    }

    pub fn tokenize_stats(&self) -> Arc<TokenizeStats> {
        Arc::clone(self.tokenizer.stats())
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // A raw prompt (`/v1/completions`) reads control-token spellings as the tokens they name,
        // as HF does: only chat turns keep the client's text out of the template's markers.
        self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// The prompt of a conversation as token ids: the template's markers as control tokens, the
    /// client's text as ordinary text. `tool_choice: none` leaves the tools out. `named` is the
    /// effort the client asked for, if it named one a level can be read from.
    pub fn build_prompt(
        &self,
        turns: &[ChatTurn<'_>],
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        thinking: bool,
        named: Option<Effort>,
    ) -> Result<Vec<u32>> {
        let tools = if matches!(tool_choice, ResolvedToolChoice::None) {
            &[][..]
        } else {
            tools
        };
        let effort = Effort::resolve_named(thinking, named);
        let prompt = render_prompt(turns, tools, effort)?;
        self.tokenizer
            .encode_prompt(&prompt)
            .map_err(|e| anyhow!("encode the prompt: {e}"))
    }

    /// [`Self::build_prompt`] for a request's own overrides.
    pub fn build_chat_input(
        &self,
        turns: &[ChatTurn<'_>],
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        thinking: bool,
        ov: &SamplingOverrides,
    ) -> Result<Vec<u32>> {
        self.build_prompt(turns, tools, tool_choice, thinking, named_effort(ov))
    }

    /// The cost of the last request.
    pub fn last_run(&self) -> RunStats {
        self.last_run
    }

    /// Forgets the retained conversation.
    pub fn clear_retained(&mut self) -> bool {
        self.retained.take().is_some()
    }

    /// One chat turn: renders the prompt, generates, and reads the reply. `on_event` gets the
    /// reply as it is read (reasoning and text deltas, the start of a tool call); an `Err` whose
    /// message is `stop sequence` from it ends the turn there, as for the other backends.
    pub fn chat(
        &mut self,
        turns: &[ChatTurn<'_>],
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        thinking: bool,
        ov: &SamplingOverrides,
        on_event: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<ParsedResponse> {
        let prompt = self.build_chat_input(turns, tools, tool_choice, thinking, ov)?;
        let mut parser = ResponseParser::new(self.markers);
        // The loop below needs `&mut self` for the caches, so the decode closure holds its own
        // handle on the tokenizer.
        let tokenizer = Arc::clone(&self.tokenizer);
        let decode = move |ids: &[u32]| -> Result<String> {
            tokenizer
                .decode(ids, false)
                .map_err(|e| anyhow!("decode: {e}"))
        };
        let sampling = sampling_config(temperature, top_p, ov);
        let mut stopped_by_sink = false;
        let generated = self.generate_with(
            &prompt,
            max_new_tokens,
            sampling,
            ov,
            &mut |token| match parser.push(token, &decode, on_event) {
                Ok(()) => Ok(parser.is_done()),
                Err(e) if e.to_string() == "stop sequence" => {
                    stopped_by_sink = true;
                    Ok(true)
                }
                Err(e) => Err(e),
            },
        )?;
        let response = if stopped_by_sink {
            // The sink asked to stop: close what is open without sending it anything more.
            parser.finish(&decode, &mut |_| Ok(()))?
        } else {
            parser.finish(&decode, on_event)?
        };
        Ok(response.stopped(StopCause::after_decode(
            &generated,
            max_new_tokens,
            &self.eos,
        )))
    }

    /// Generates from a token prompt and returns the generated ids, the end-of-turn marker
    /// included when the model wrote it (`/v1/completions`).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &SamplingOverrides,
    ) -> Result<Vec<u32>> {
        let sampling = sampling_config(temperature, top_p, ov);
        let eos = self.eos.clone();
        self.generate_with(prompt, max_new_tokens, sampling, ov, &mut |t| {
            Ok(eos.contains(&t))
        })
    }

    /// The generation loop. `on_token` sees each generated token and says whether to stop (the
    /// end-of-turn marker is a token like the others and is passed to it too).
    fn generate_with(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampling: Option<SamplingConfig>,
        ov: &SamplingOverrides,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "an empty prompt");
        ensure!(
            prompt.len() + max_new_tokens.min(1) <= self.max_context,
            "a prompt of {} tokens is past this backend's context of {}",
            prompt.len(),
            self.max_context
        );
        let (mut cache, mut points, start) = self.take_reusable(prompt);
        let began = std::time::Instant::now();
        let logits = self.prefill(prompt, start, &mut cache, &mut points)?;
        let prefill_ms = began.elapsed().as_secs_f64() * 1e3;
        let decode_began = std::time::Instant::now();

        let mut rng = Xorshift64::new(crate::resolve_sampling_seed(ov));
        let mut seen: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut logits = logits;
        while generated.len() < max_new_tokens {
            let token = pick(&logits, &seen, sampling.as_ref(), &mut rng)?;
            generated.push(token);
            let stop = on_token(token)?;
            if self.eos.contains(&token) {
                // The end of the turn is part of what a reply read back will contain.
                self.model.advance(&[token], &mut cache)?;
                seen.push(token);
                break;
            }
            if stop || generated.len() >= max_new_tokens {
                break;
            }
            logits = self.model.forward(&[token], &mut cache, true)?;
            logits.eval()?;
            seen.push(token);
        }
        points.push((cache.len(), cache.conv_snapshot()?));
        self.last_run = RunStats {
            prompt_tokens: prompt.len(),
            reused_tokens: start,
            prefill_ms,
            generated_tokens: generated.len(),
            decode_ms: decode_began.elapsed().as_secs_f64() * 1e3,
        };
        self.retained = Some(Retained {
            tokens: seen,
            cache,
            points,
        });
        Ok(generated)
    }

    /// The retained caches taken back to the longest point `prompt` shares with them, or fresh
    /// caches. Returns the caches, the points still valid, and how many prompt tokens they hold.
    fn take_reusable(
        &mut self,
        prompt: &[u32],
    ) -> (Inkling2Cache, Vec<(usize, ConvSnapshot)>, usize) {
        let fresh = |model: &Inkling2Model| (model.make_cache(), Vec::new(), 0);
        let Some(mut kept) = self.retained.take() else {
            return fresh(&self.model);
        };
        let Some(len) = reusable_point(
            kept.points.iter().map(|(len, _)| *len),
            &kept.tokens,
            prompt,
        ) else {
            return fresh(&self.model);
        };
        let at = kept
            .points
            .iter()
            .position(|(l, _)| *l == len)
            .expect("the point was found in this list");
        if kept.cache.rewind(len, &kept.points[at].1).is_err() {
            return fresh(&self.model);
        }
        kept.points.truncate(at + 1);
        (kept.cache, kept.points, len)
    }

    /// Feeds `prompt[start..]` and returns the logits of its last token. Records the points a
    /// later prompt can start from: one token before the end, and the end.
    fn prefill(
        &self,
        prompt: &[u32],
        start: usize,
        cache: &mut Inkling2Cache,
        points: &mut Vec<(usize, ConvSnapshot)>,
    ) -> Result<Array> {
        let last = prompt.len() - 1;
        if last > start {
            for chunk in prompt[start..last].chunks(PREFILL_CHUNK) {
                self.model.advance(chunk, cache)?;
            }
            points.push((last, cache.conv_snapshot()?));
        }
        let logits = self.model.forward(&prompt[last..], cache, true)?;
        logits.eval()?;
        cache.eval_state()?;
        points.push((prompt.len(), cache.conv_snapshot()?));
        Ok(logits)
    }
}

/// The next token from `[1, 1, vocab]` logits: the largest, or a draw under `sampling`.
fn pick(
    logits: &Array,
    seen: &[u32],
    sampling: Option<&SamplingConfig>,
    rng: &mut Xorshift64,
) -> Result<u32> {
    match sampling {
        None => {
            let flat = mlx_rs::ops::reshape(logits, &[-1])?;
            let best = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false)?;
            best.eval()?;
            Ok(best.item::<u32>())
        }
        Some(cfg) => {
            let mut host = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(logits)?;
            Ok(sample_from_logits(&mut host, seen, cfg, rng))
        }
    }
}

/// The effort a request names: its `reasoning_effort` as spelled, when a level can be read from
/// it.
pub fn named_effort(ov: &SamplingOverrides) -> Option<Effort> {
    ov.reasoning_effort_raw.as_deref().and_then(Effort::parse)
}

/// The sampling of a request. `None` is greedy.
///
/// The checkpoint's `generation_config.json` samples at temperature 1; the server's default of 0.7
/// means the client sent none, so it becomes 1 as for Gemma 4. A top-k of 64 bounds the candidates
/// so that a tail token cannot be drawn on 4-bit weights; the repeat penalty is off unless the
/// request sets one, since the model card names none.
fn sampling_config(temperature: f32, top_p: f32, ov: &SamplingOverrides) -> Option<SamplingConfig> {
    let temperature = if (temperature - 0.7).abs() < 1e-6 {
        1.0
    } else {
        temperature
    };
    let cfg = SamplingConfig {
        temperature,
        top_p,
        top_k: ov.top_k.unwrap_or(64),
        repeat_penalty: ov.repeat_penalty.unwrap_or(1.0),
        repeat_penalty_last_n: 64,
        presence_penalty: ov.presence_penalty.unwrap_or(0.0),
        frequency_penalty: ov.frequency_penalty.unwrap_or(0.0),
        min_p: ov.min_p.unwrap_or(0.0),
        seed: crate::resolve_sampling_seed(ov),
        dry: lumen_core::dry::dry_config_from_env(),
    };
    if cfg.is_greedy() { None } else { Some(cfg) }
}

/// A chat turn from the flat `(role, content)` pairs the older entry points carry.
pub fn turns_from_pairs(messages: &[(String, String)]) -> Result<Vec<ChatTurn<'_>>> {
    messages
        .iter()
        .map(|(role, content)| match role.as_str() {
            "system" => Ok(ChatTurn::System(content)),
            "user" => Ok(ChatTurn::User(content)),
            "assistant" => Ok(ChatTurn::Assistant {
                text: content,
                tool_calls: &[],
            }),
            other => Err(crate::chat_io::UnsupportedRequest(format!(
                "Inkling 2 has no message role {other:?} (system, user, assistant and tool)"
            ))
            .into()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_reusable_when_its_tokens_are_a_proper_prefix_of_the_new_prompt() {
        let seen = [1, 2, 3, 4, 5, 6];
        let lens = || [2usize, 4, 6].into_iter();
        // The longest shared point below the new prompt's length.
        assert_eq!(
            reusable_point(lens(), &seen, &[1, 2, 3, 4, 9, 9, 9]),
            Some(4)
        );
        assert_eq!(
            reusable_point(lens(), &seen, &[1, 2, 3, 4, 5, 6, 7]),
            Some(6)
        );
        // A prompt that is exactly a point leaves nothing to feed: the point before it is used.
        assert_eq!(reusable_point(lens(), &seen, &[1, 2, 3, 4, 5, 6]), Some(4));
        // A point whose tokens differ is not reusable, however long.
        assert_eq!(reusable_point(lens(), &seen, &[1, 9, 3, 4, 5, 6, 7]), None);
        assert_eq!(reusable_point(lens(), &seen, &[1, 2, 9, 9, 9]), Some(2));
        // A prompt shorter than every point, or nothing recorded.
        assert_eq!(reusable_point(lens(), &seen, &[1, 2]), None);
        assert_eq!(reusable_point(std::iter::empty(), &seen, &[1, 2, 3]), None);
        // A point past what was seen is a stale record.
        assert_eq!(reusable_point([9usize].into_iter(), &seen, &[1; 20]), None);
    }

    #[test]
    fn flat_messages_become_turns() {
        let messages = vec![
            ("system".to_string(), "s".to_string()),
            ("user".to_string(), "u".to_string()),
            ("assistant".to_string(), "a".to_string()),
        ];
        let turns = turns_from_pairs(&messages).unwrap();
        assert!(matches!(turns[0], ChatTurn::System("s")));
        assert!(matches!(turns[1], ChatTurn::User("u")));
        assert!(matches!(turns[2], ChatTurn::Assistant { text: "a", .. }));
        let bad = vec![("tool".to_string(), "t".to_string())];
        assert!(turns_from_pairs(&bad).is_err());
    }

    #[test]
    fn a_temperature_of_the_servers_default_is_the_models_and_zero_is_greedy() {
        let ov = SamplingOverrides::default();
        let cfg = sampling_config(0.7, 1.0, &ov).expect("sampled");
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, 64);
        assert!(sampling_config(0.0, 1.0, &ov).is_none());
        let ov = SamplingOverrides {
            top_k: Some(5),
            ..Default::default()
        };
        assert_eq!(sampling_config(0.3, 0.9, &ov).unwrap().top_k, 5);
    }
}

/// The backend on the real checkpoint. Set `LUMEN_INKLING2_DIR` to the checkpoint directory:
///
/// ```text
/// LUMEN_INKLING2_DIR=~/.slotstream/models/Inkling-2-Mini-oQ4e-mtp \
///   cargo test -p lumen-mlx --features mlx-native --lib inkling2::backend -- --ignored --test-threads=1
/// ```
#[cfg(test)]
mod real_checkpoint_tests {
    use super::*;

    fn backend() -> Inkling2Backend {
        let dir = std::env::var("LUMEN_INKLING2_DIR").expect("LUMEN_INKLING2_DIR");
        Inkling2Backend::from_dir("inkling2", Path::new(&dir)).expect("load")
    }

    /// One greedy turn over `turns`, returning what the model said.
    fn greedy(b: &mut Inkling2Backend, turns: &[ChatTurn<'_>], max_new: usize) -> ParsedResponse {
        let ov = SamplingOverrides::default();
        b.chat(
            turns,
            &[],
            &ResolvedToolChoice::Auto,
            max_new,
            0.0,
            1.0,
            false,
            &ov,
            &mut |_| Ok(()),
        )
        .expect("chat")
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_DIR"]
    fn a_repeated_prompt_is_fed_only_its_last_token_and_says_the_same() {
        let mut b = backend();
        let turns = [ChatTurn::User("Name one primary color. One word.")];
        let first = greedy(&mut b, &turns, 48);
        let cold = b.last_run();
        assert_eq!(cold.reused_tokens, 0);
        let second = greedy(&mut b, &turns, 48);
        let warm = b.last_run();
        assert_eq!(warm.reused_tokens, warm.prompt_tokens - 1, "{warm:?}");
        assert_eq!(second.visible, first.visible);
        assert_eq!(second.reasoning, first.reasoning);
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_DIR"]
    fn a_turn_continued_from_the_retained_caches_says_what_a_cold_prefill_says() {
        let mut b = backend();
        let first = [ChatTurn::User(
            "What is the capital of France? One short sentence.",
        )];
        let reply = greedy(&mut b, &first, 64);
        let said = if reply.reasoning.is_empty() {
            reply.visible.clone()
        } else {
            crate::chat_io::join_reasoning_envelope(&reply.reasoning, &reply.visible)
        };
        let second = [
            ChatTurn::User("What is the capital of France? One short sentence."),
            ChatTurn::Assistant {
                text: &said,
                tool_calls: &[],
            },
            ChatTurn::User("And Japan? One short sentence."),
        ];
        let warm = greedy(&mut b, &second, 64);
        let warm_stats = b.last_run();
        assert!(warm_stats.reused_tokens > 0, "{warm_stats:?}");
        assert!(
            warm_stats.reused_tokens >= warm_stats.prompt_tokens - 12,
            "the reply read back must start from where the model stopped: {warm_stats:?}"
        );
        b.clear_retained();
        let cold = greedy(&mut b, &second, 64);
        assert_eq!(b.last_run().reused_tokens, 0);
        // The two runs feed the same tokens in different chunkings, so MLX may pick other kernels
        // for them: the text has to agree, which it does unless a near-tie flips.
        assert_eq!(warm.visible, cold.visible, "warm {warm_stats:?}");
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_DIR"]
    fn an_edited_history_falls_back_to_a_cold_prefill() {
        let mut b = backend();
        greedy(&mut b, &[ChatTurn::User("Say hello.")], 16);
        let edited = [ChatTurn::User("Say goodbye.")];
        greedy(&mut b, &edited, 16);
        assert_eq!(b.last_run().reused_tokens, 0);
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_DIR"]
    fn a_stop_from_the_sink_ends_the_turn_and_is_not_an_error() {
        let mut b = backend();
        let turns = [ChatTurn::User(
            "Count from one to twenty, in words, separated by commas.",
        )];
        let ov = SamplingOverrides::default();
        let mut pieces = 0;
        let reply = b
            .chat(
                &turns,
                &[],
                &ResolvedToolChoice::Auto,
                200,
                0.0,
                1.0,
                false,
                &ov,
                &mut |_| {
                    pieces += 1;
                    if pieces >= 3 {
                        Err(anyhow!("stop sequence"))
                    } else {
                        Ok(())
                    }
                },
            )
            .expect("a stop sequence is not an error");
        assert!(pieces >= 3);
        assert!(b.last_run().generated_tokens < 200);
        let _ = reply;
    }
}
