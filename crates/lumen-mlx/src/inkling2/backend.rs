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
//!
//! Images ride along. A prompt names each image by a run of image tokens (one per 40-pixel block);
//! what the run stands for is the image's features, computed by the vision tower only when a token
//! of the run has to be fed. So that a prompt whose caches are retained is not mistaken for another
//! with the same grid but a different picture, each image token's key carries a hash of the image,
//! and a follow-up turn that sends the same image back reuses the caches without decoding it.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use lumen_core::sampling::{
    SamplingConfig, Xorshift64, sample_distribution, sample_from_logits, sampling_distribution,
    speculative_accept,
};
use mlx_rs::Array;
use mlx_rs::ops::indexing::TryIndexOp;

use super::chat::{Effort, IMAGE_TOKEN, render_prompt_with_images};
use super::draft::{DraftModel, TapRing};
use super::model::{ConvSnapshot, ImageRows, Inkling2Cache, Inkling2Model, LoadOptions};
use super::pixels::{prepare, token_count};
use super::response::{Markers, ResponseParser};
use super::vision::VisionTower;
use crate::SamplingOverrides;
use crate::chat_io::{
    BackendStreamEvent, ChatTurn, ParsedResponse, ResolvedToolChoice, StopCause, ToolDef,
    tool_result_order,
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

lumen_flags::flag! {
    /// Speculative decoding for Inkling 2: the checkpoint's bundled DSpark draft head
    /// (`mtp.safetensors`) proposes a block of tokens and the model checks the block in one pass.
    /// A greedy request keeps the tokens the model would have chosen itself; a sampled one accepts
    /// each draft with the probability the request's own sampling gives it and draws the
    /// replacement from what is left, which is exactly the distribution plain sampling draws from.
    /// The number of drafts checked per block follows how often recent drafts agreed, so text the
    /// draft cannot predict does not run slower than the plain loop.
    ///
    /// Measured on an M3 Max (Inkling 2 Mini, 150 tokens): 25 tokens/s plain. Greedy, over eight
    /// workloads: 26 (a poem, a translation), 29 (prose), 31 (JSON), 32 (code), 35 (random words),
    /// 37 (reasoning), 53 (a list of primes), none slower. Sampled at the default temperature:
    /// 25 (prose, a poem), 26 (code), 48 (a list of primes). It costs 0.7 GiB for the draft head,
    /// loaded with the first request (about a second). A pass over a block does not round exactly
    /// like single-token steps in bfloat16, so a greedy continuation can differ from the plain
    /// loop's where the two best tokens are within a bfloat16 step or two of each other (about 1
    /// position in 200 here); hence a behavior flag. `=0` runs the plain loop.
    pub(crate) inkling2_draft {
        env: "LUMEN_INKLING2_DRAFT",
        default: true,
        kind: Behavior,
    }
}

/// Drafts a block proposes and the model checks, when the flag turns speculation on.
const DEFAULT_DRAFT_DEPTH: usize = 7;

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
    /// Images the vision tower encoded for this request (those whose tokens were fed), and how
    /// long that took, the tower's first load included.
    pub images_encoded: usize,
    pub image_ms: f64,
    /// The speculative loop's work, all zero when the request ran the plain loop.
    pub spec: SpecStats,
}

/// What the speculative decode loop did in one request.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecStats {
    /// Draft blocks proposed and checked.
    pub cycles: usize,
    /// Drafts put to the check, and how many of them the model agreed with.
    pub drafted: usize,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
}

impl SpecStats {
    /// Tokens a cycle gives on average: the model's own next token plus the drafts it agreed with.
    pub fn tokens_per_cycle(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            (self.cycles + self.accepted) as f64 / self.cycles as f64
        }
    }
}

/// What a token of a prompt is, for telling prompts apart: the token, and for an image token a hash
/// of the image its run stands for (0 for every other token).
type Key = (u32, u64);

/// A prompt: its token ids and which runs of them are images.
struct Prompt {
    ids: Vec<u32>,
    keys: Vec<Key>,
    /// `(start, len)` of each image's run of image tokens, in prompt order.
    runs: Vec<(usize, usize)>,
}

impl Prompt {
    /// A prompt of plain tokens.
    fn plain(ids: &[u32]) -> Self {
        Self {
            ids: ids.to_vec(),
            keys: ids.iter().map(|&t| (t, 0)).collect(),
            runs: Vec::new(),
        }
    }
}

/// The maximal runs of `image_token` in `ids`, as `(start, len)`.
fn image_runs(ids: &[u32], image_token: u32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;
    for (i, &t) in ids.iter().enumerate() {
        match (t == image_token, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                runs.push((s, i - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push((s, ids.len() - s));
    }
    runs
}

fn image_hash(encoded: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    encoded.hash(&mut hasher);
    hasher.finish()
}

/// A conversation's caches, to be picked up by the next request.
struct Retained {
    /// What the caches have seen.
    keys: Vec<Key>,
    cache: Inkling2Cache,
    /// Lengths the caches can go back to, ascending, with their convolution states.
    points: Vec<(usize, ConvSnapshot)>,
    /// The draft head's context taps of the newest positions, when the request that left these
    /// caches speculated; it ends where the caches do.
    taps: Option<TapRing>,
}

/// What a request starts from: caches (retained or fresh), the points they can go back to, how
/// many prompt tokens they hold, and the taps that go with them.
struct Resume {
    cache: Inkling2Cache,
    points: Vec<(usize, ConvSnapshot)>,
    start: usize,
    taps: Option<TapRing>,
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
    dir: PathBuf,
    /// The id of the image token, when the checkpoint has a vision tower the prompt can name.
    image_token: Option<u32>,
    /// The vision tower, read from the checkpoint the first time an image has to be encoded.
    tower: Option<VisionTower>,
    /// The draft head, read the first time speculation is asked for; `Some(None)` once it is
    /// known the checkpoint has none.
    draft: Option<Option<DraftModel>>,
    /// Drafts to check per block; `None` or 0 is the plain loop.
    draft_depth: Option<usize>,
}

/// The longest recorded point the caches can go back to for a prompt `new`: its tokens are a
/// prefix of `new` and shorter than it (at least one token is left to feed, since the logits of
/// the next token come from feeding).
fn reusable_point<T: PartialEq>(
    lengths: impl Iterator<Item = usize>,
    seen: &[T],
    new: &[T],
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
        // The image token is the one the processor config names and the renderer writes.
        let processor: Option<serde_json::Value> =
            std::fs::read_to_string(dir.join("processor_config.json"))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
        let names_it = processor
            .as_ref()
            .and_then(|p| p["image_token"].as_str())
            .is_some_and(|token| token == IMAGE_TOKEN);
        let image_token = (names_it && config.get("vision_config").is_some())
            .then(|| tokenizer.single_token_id(IMAGE_TOKEN))
            .flatten();
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
            dir: dir.to_path_buf(),
            image_token,
            tower: None,
            draft: None,
            draft_depth: inkling2_draft::get().then_some(DEFAULT_DRAFT_DEPTH),
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

    /// Token ids as text for `/v1/completions`: the control tokens (the end of the turn among
    /// them) are left out, as the other backends do.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
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
        let (turns, _) = in_call_order(turns, &[]);
        Ok(self
            .prompt_with_images(&turns, &[], tools, tool_choice, thinking, named)?
            .ids)
    }

    /// [`Self::build_prompt`] for turns that carry images (`images[i]` are the encoded images of
    /// `turns[i]`): each image is a run of image tokens, one per block, found again in the ids so
    /// the features can be put where they belong. Only the images' headers are read here.
    fn prompt_with_images(
        &self,
        turns: &[ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        thinking: bool,
        named: Option<Effort>,
    ) -> Result<Prompt> {
        let tools = if matches!(tool_choice, ResolvedToolChoice::None) {
            &[][..]
        } else {
            tools
        };
        let effort = Effort::resolve_named(thinking, named);
        let attached = images.iter().any(|per_turn| !per_turn.is_empty());
        if !attached {
            let prompt = render_prompt_with_images(turns, tools, effort, &[])?;
            let ids = self
                .tokenizer
                .encode_prompt(&prompt)
                .map_err(|e| anyhow!("encode the prompt: {e}"))?;
            return Ok(Prompt::plain(&ids));
        }
        let unsupported = |why: String| crate::chat_io::UnsupportedRequest(why);
        let Some(image_token) = self.image_token else {
            return Err(unsupported(
                "this Inkling 2 checkpoint has no vision tower, so it cannot read images".into(),
            )
            .into());
        };
        let mut counts: Vec<Vec<usize>> = Vec::with_capacity(turns.len());
        for (index, turn) in turns.iter().enumerate() {
            let own = images.get(index).map_or(&[][..], Vec::as_slice);
            if !own.is_empty() && !matches!(turn, ChatTurn::User(_) | ChatTurn::Tool { .. }) {
                return Err(unsupported(
                    "images are only supported in user and tool messages".into(),
                )
                .into());
            }
            counts.push(
                own.iter()
                    .enumerate()
                    .map(|(n, bytes)| {
                        token_count(bytes).map_err(|e| {
                            anyhow::Error::from(unsupported(format!(
                                "image {} of message {index} cannot be read: {e}",
                                n + 1
                            )))
                        })
                    })
                    .collect::<Result<_>>()?,
            );
        }
        let text = render_prompt_with_images(turns, tools, effort, &counts)?;
        let ids = self
            .tokenizer
            .encode_prompt(&text)
            .map_err(|e| anyhow!("encode the prompt: {e}"))?;
        let runs = image_runs(&ids, image_token);
        let flat: Vec<(usize, &Vec<u8>)> = counts
            .iter()
            .flatten()
            .copied()
            .zip(images.iter().flatten())
            .collect();
        ensure!(
            runs.len() == flat.len() && runs.iter().zip(&flat).all(|(r, f)| r.1 == f.0),
            "the prompt's image tokens do not match the images it carries"
        );
        let mut keys: Vec<Key> = ids.iter().map(|&t| (t, 0)).collect();
        for (&(start, len), (_, bytes)) in runs.iter().zip(&flat) {
            let hash = image_hash(bytes);
            for key in &mut keys[start..start + len] {
                key.1 = hash;
            }
        }
        Ok(Prompt { ids, keys, runs })
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

    /// Sets how many drafts the speculative loop checks per block (`None` or 0: the plain loop),
    /// whatever the flag said at load. For measuring both in one process.
    pub fn set_draft_depth(&mut self, depth: Option<usize>) {
        self.draft_depth = depth;
    }

    pub fn draft_depth(&self) -> Option<usize> {
        self.draft_depth
    }

    /// The draft head, read from the checkpoint the first time it is needed. Whether the
    /// checkpoint has one.
    fn ensure_draft(&mut self) -> Result<bool> {
        if self.draft.is_none() {
            let loaded = DraftModel::load(&self.dir, &self.model)
                .with_context(|| format!("load the draft head from {}", self.dir.display()))?;
            if let Some(d) = &loaded {
                let visits = self.model.num_visits();
                ensure!(
                    d.args().target_layer_ids.iter().all(|&id| id <= visits),
                    "the draft reads layer visits {:?}, the model has {visits}",
                    d.args().target_layer_ids
                );
            }
            self.draft = Some(loaded);
        }
        Ok(matches!(self.draft, Some(Some(_))))
    }

    /// Whether the draft head can be used. One that cannot be read (a damaged or mismatched
    /// `mtp.safetensors`) is an accelerator lost, not a reason to fail every request: it is said
    /// once and the plain loop serves from then on.
    fn draft_ready(&mut self) -> bool {
        match self.ensure_draft() {
            Ok(ready) => ready,
            Err(e) => {
                eprintln!("[inkling2] the draft head is unusable, decoding without it: {e:#}");
                self.draft = Some(None);
                false
            }
        }
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
    /// reply as it is read (reasoning and text deltas, the start of a tool call); an `Err` from it
    /// ends the turn there (a stop sequence matched, or the client is gone), as for the other
    /// backends.
    pub fn chat(
        &mut self,
        turns: &[ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        thinking: bool,
        ov: &SamplingOverrides,
        on_event: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<ParsedResponse> {
        let (turns, images) = in_call_order(turns, images);
        let prompt = self.prompt_with_images(
            &turns,
            &images,
            tools,
            tool_choice,
            thinking,
            named_effort(ov),
        )?;
        let max_new_tokens = reply_budget(self.max_context, prompt.ids.len(), max_new_tokens);
        let sources: Vec<&[u8]> = images.iter().flatten().map(Vec::as_slice).collect();
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
        // Any error the sink returns ends the turn: it has what it needs (a stop sequence matched)
        // or the client is gone, and the server tells those from a failure by its own record, never
        // by the message. Only the sink's errors do: the parser's own (a decode failure) are real.
        let sink_failed = std::cell::Cell::new(false);
        let mut stopped_by_sink = false;
        let generated = self.generate_with(
            &prompt,
            &sources,
            max_new_tokens,
            sampling,
            ov,
            &mut |token| {
                let pushed = parser.push(token, &decode, &mut |event| {
                    on_event(event).inspect_err(|_| sink_failed.set(true))
                });
                match pushed {
                    Ok(()) => Ok(parser.is_done()),
                    Err(_) if sink_failed.get() => {
                        stopped_by_sink = true;
                        Ok(true)
                    }
                    Err(e) => Err(e),
                }
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
        self.generate_with(
            &Prompt::plain(prompt),
            &[],
            reply_budget(self.max_context, prompt.len(), max_new_tokens),
            sampling,
            ov,
            &mut |t| Ok(eos.contains(&t)),
        )
    }

    /// The generation loop. `on_token` sees each generated token and says whether to stop (the
    /// end-of-turn marker is a token like the others and is passed to it too).
    fn generate_with(
        &mut self,
        prompt: &Prompt,
        sources: &[&[u8]],
        max_new_tokens: usize,
        sampling: Option<SamplingConfig>,
        ov: &SamplingOverrides,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<Vec<u32>> {
        ensure!(!prompt.ids.is_empty(), "an empty prompt");
        ensure!(
            prompt.ids.len() < self.max_context,
            "a prompt of {} tokens leaves no room to reply in this backend's context of {}",
            prompt.ids.len(),
            self.max_context
        );
        let speculate = self.draft_depth.unwrap_or(0) > 0 && self.draft_ready();
        let began = std::time::Instant::now();
        // The features come first, before the retained conversation is taken: an image that turns
        // out unreadable is the client's error, and must not cost it the caches.
        let hint = self.reusable_start(&prompt.keys);
        let image_began = std::time::Instant::now();
        let mut rows = self.image_rows(prompt, sources, hint)?;
        let Resume {
            mut cache,
            mut points,
            start,
            taps,
        } = self.take_reusable(&prompt.keys);
        if start != hint {
            // The caches could not be taken back after all: the images before `start` are fed too.
            rows = self.image_rows(prompt, sources, start)?;
        }
        let image_ms = image_began.elapsed().as_secs_f64() * 1e3;
        let images_encoded = prompt
            .runs
            .iter()
            .filter(|&&(at, len)| at + len > start)
            .count();
        let mut ring = taps
            .filter(|_| speculate)
            .unwrap_or_else(|| TapRing::starting_at(start));
        let logits = if speculate {
            self.prefill_taps(
                &prompt.ids,
                start,
                &mut cache,
                &mut points,
                rows.as_ref(),
                &mut ring,
            )?
        } else {
            self.prefill(&prompt.ids, start, &mut cache, &mut points, rows.as_ref())?
        };
        let prefill_ms = began.elapsed().as_secs_f64() * 1e3;
        let decode_began = std::time::Instant::now();

        let mut rng = Xorshift64::new(crate::resolve_sampling_seed(ov));
        let mut seen: Vec<u32> = prompt.ids.clone();
        let mut generated: Vec<u32> = Vec::new();
        let mut spec = SpecStats::default();
        if speculate {
            let draft = self
                .draft
                .as_ref()
                .and_then(Option::as_ref)
                .context("no draft head")?;
            let mut context = draft.context_from(&ring)?;
            let plan = SpecPlan {
                model: &self.model,
                tap_ids: &draft.args().target_layer_ids,
                keep: draft.args().context_rows,
                eos: &self.eos,
                max_new_tokens,
                depth: self
                    .draft_depth
                    .unwrap_or(0)
                    .min(draft.args().drafts_per_block()),
                adaptive: true,
            };
            run_speculative(
                &plan,
                &mut cache,
                &mut ring,
                &mut seen,
                &mut generated,
                &logits,
                on_token,
                &mut spec,
                sampling.as_ref(),
                &mut rng,
                &mut |anchor, rows, want, _| {
                    if want == 0 {
                        if let Some(rows) = rows {
                            draft.extend(&mut context, rows)?;
                        }
                        return Ok(Vec::new());
                    }
                    Ok(draft
                        .draft_up_to(&mut context, rows, anchor, &self.model, want)?
                        .tokens)
                },
            )?;
        } else {
            run_plain(
                &self.model,
                &self.eos,
                &mut cache,
                &mut seen,
                &mut generated,
                logits,
                max_new_tokens,
                sampling.as_ref(),
                &mut rng,
                on_token,
            )?;
        }
        points.push((cache.len(), cache.conv_snapshot()?));
        self.last_run = RunStats {
            prompt_tokens: prompt.ids.len(),
            reused_tokens: start,
            prefill_ms,
            generated_tokens: generated.len(),
            decode_ms: decode_began.elapsed().as_secs_f64() * 1e3,
            images_encoded,
            image_ms,
            spec,
        };
        let mut keys = prompt.keys.clone();
        keys.extend(seen[prompt.ids.len()..].iter().map(|&t| (t, 0)));
        if speculate {
            ring.eval()?;
        }
        self.retained = Some(Retained {
            keys,
            cache,
            points,
            taps: speculate.then_some(ring),
        });
        Ok(generated)
    }

    /// The vision tower, read from the checkpoint on first use.
    fn tower(&mut self) -> Result<&VisionTower> {
        if self.tower.is_none() {
            self.tower =
                Some(VisionTower::load(&self.dir).with_context(|| {
                    format!("load the vision tower from {}", self.dir.display())
                })?);
        }
        Ok(self.tower.as_ref().expect("loaded above"))
    }

    /// The features of the images whose tokens `prompt[start..]` contains, as the rows that stand
    /// in for those tokens; `None` when there are none (a text prompt, or every image already in
    /// the caches). `sources` are the encoded images in the order the prompt names them.
    fn image_rows(
        &mut self,
        prompt: &Prompt,
        sources: &[&[u8]],
        start: usize,
    ) -> Result<Option<ImageRows>> {
        ensure!(
            sources.len() == prompt.runs.len(),
            "{} images for {} image runs",
            sources.len(),
            prompt.runs.len()
        );
        let needed: Vec<usize> = (0..prompt.runs.len())
            .filter(|&i| prompt.runs[i].0 + prompt.runs[i].1 > start)
            .collect();
        if needed.is_empty() {
            return Ok(None);
        }
        let mut slots = vec![-1i32; prompt.ids.len()];
        let mut features: Vec<Array> = Vec::with_capacity(needed.len());
        let mut rows = 0usize;
        for &i in &needed {
            let (at, len) = prompt.runs[i];
            let image = prepare(sources[i]).map_err(|e| {
                crate::chat_io::UnsupportedRequest(format!("image {} cannot be read: {e}", i + 1))
            })?;
            ensure!(
                image.tokens() == len,
                "image {} is {} blocks once decoded, its header said {len}",
                i + 1,
                image.tokens()
            );
            let encoded = self.tower()?.encode(&image)?;
            encoded.eval()?;
            for (offset, slot) in slots[at..at + len].iter_mut().enumerate() {
                *slot = (rows + offset) as i32;
            }
            rows += len;
            features.push(encoded);
        }
        let refs: Vec<&Array> = features.iter().collect();
        let all = if refs.len() == 1 {
            features[0].clone()
        } else {
            mlx_rs::ops::concatenate_axis(&refs, 0)?
        };
        Ok(Some(ImageRows::new(slots, all)?))
    }

    /// How many leading tokens of `keys` the retained caches can supply: where
    /// [`Self::take_reusable`] will take them back to, without taking them.
    fn reusable_start(&self, keys: &[Key]) -> usize {
        self.retained
            .as_ref()
            .and_then(|kept| {
                reusable_point(kept.points.iter().map(|(len, _)| *len), &kept.keys, keys)
            })
            .unwrap_or(0)
    }

    /// The retained caches taken back to the longest point `prompt` shares with them, or fresh
    /// caches, with the points still valid, how many prompt tokens they hold and their taps.
    fn take_reusable(&mut self, prompt: &[Key]) -> Resume {
        let fresh = |model: &Inkling2Model| Resume {
            cache: model.make_cache(),
            points: Vec::new(),
            start: 0,
            taps: None,
        };
        let Some(mut kept) = self.retained.take() else {
            return fresh(&self.model);
        };
        let Some(len) = reusable_point(kept.points.iter().map(|(len, _)| *len), &kept.keys, prompt)
        else {
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
        // The taps go back with the caches; if they cannot, the draft starts from what is fed next.
        let taps = kept
            .taps
            .and_then(|mut t| t.truncate_to(len).ok().map(|()| t));
        Resume {
            cache: kept.cache,
            points: kept.points,
            start: len,
            taps,
        }
    }

    /// Feeds `prompt[start..]` and returns the logits of its last token. Records the points a
    /// later prompt can start from: one token before the end, and the end.
    fn prefill(
        &self,
        prompt: &[u32],
        start: usize,
        cache: &mut Inkling2Cache,
        points: &mut Vec<(usize, ConvSnapshot)>,
        images: Option<&ImageRows>,
    ) -> Result<Array> {
        let last = prompt.len() - 1;
        if last > start {
            for chunk in prompt[start..last].chunks(PREFILL_CHUNK) {
                self.model.advance_with(chunk, cache, images)?;
            }
            points.push((last, cache.conv_snapshot()?));
        }
        let logits = self
            .model
            .forward_with(&prompt[last..], cache, true, images)?;
        logits.eval()?;
        cache.eval_state()?;
        points.push((prompt.len(), cache.conv_snapshot()?));
        Ok(logits)
    }
}

impl Inkling2Backend {
    /// [`Self::prefill`] that also keeps the draft head's taps of the fed positions in `ring`.
    fn prefill_taps(
        &self,
        prompt: &[u32],
        start: usize,
        cache: &mut Inkling2Cache,
        points: &mut Vec<(usize, ConvSnapshot)>,
        images: Option<&ImageRows>,
        ring: &mut TapRing,
    ) -> Result<Array> {
        let draft = self
            .draft
            .as_ref()
            .and_then(Option::as_ref)
            .context("no draft head")?;
        let (tap_ids, keep) = (&draft.args().target_layer_ids, draft.args().context_rows);
        let last = prompt.len() - 1;
        if last > start {
            for chunk in prompt[start..last].chunks(PREFILL_CHUNK) {
                let taps = self.model.advance_taps(chunk, cache, tap_ids, images)?;
                ring.push(&taps, keep)?;
            }
            points.push((last, cache.conv_snapshot()?));
        }
        let (logits, taps) =
            self.model
                .forward_taps(&prompt[last..], cache, true, tap_ids, images, false)?;
        logits.eval()?;
        taps.eval()?;
        cache.eval_state()?;
        ring.push(&taps, keep)?;
        points.push((prompt.len(), cache.conv_snapshot()?));
        Ok(logits)
    }
}

/// The tokens a block commits when the request samples: `rows` are the model's logits for the
/// anchor and the `block` of drafts (`block.len() + 1` rows of `vocab`), `recent` the tokens fed
/// before the anchor (with the anchor and the drafts accepted so far, what the penalties look at). Each draft is accepted with the
/// probability the model's own sampling distribution gives it, and a rejected one is replaced by a
/// draw from that distribution without it (Leviathan and Chen with a draft distribution that
/// puts all its mass on the draft token, which is what an argmax draft is); after the last accepted
/// draft the model's own distribution gives the next token. The tokens committed are distributed
/// exactly as `block.len() + 1` steps of plain sampling would draw them. The last token is the
/// model's own, so `committed.len() - 1` drafts were accepted.
fn sampled_block(
    rows: &[f32],
    vocab: usize,
    block: &[u32],
    recent: &[u32],
    anchor: u32,
    cfg: &SamplingConfig,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let mut history = Vec::with_capacity(recent.len() + 1 + block.len());
    history.extend_from_slice(recent);
    history.push(anchor);
    let mut committed = Vec::with_capacity(block.len() + 1);
    for i in 0..=block.len() {
        let mut row = rows[i * vocab..(i + 1) * vocab].to_vec();
        let p = sampling_distribution(&mut row, &history, cfg);
        let Some(&draft) = block.get(i) else {
            committed.push(sample_distribution(&p, rng));
            break;
        };
        let mut q = vec![0.0f32; vocab];
        q[draft as usize] = 1.0;
        let verdict = speculative_accept(&p, &q, draft, rng);
        committed.push(verdict.token);
        if !verdict.accepted {
            break;
        }
        history.push(draft);
    }
    committed
}

/// The plain decode loop: the next token from the logits, fed back, until the end-of-turn marker
/// (fed too, so a reply read back continues from it), a stop from `on_token`, or the limit.
#[allow(clippy::too_many_arguments)]
fn run_plain(
    model: &Inkling2Model,
    eos: &[u32],
    cache: &mut Inkling2Cache,
    seen: &mut Vec<u32>,
    generated: &mut Vec<u32>,
    mut logits: Array,
    max_new_tokens: usize,
    sampling: Option<&SamplingConfig>,
    rng: &mut Xorshift64,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
) -> Result<()> {
    while generated.len() < max_new_tokens {
        let token = pick(&logits, seen, sampling, rng)?;
        generated.push(token);
        let stop = on_token(token)?;
        if eos.contains(&token) {
            model.advance(&[token], cache)?;
            seen.push(token);
            break;
        }
        if stop || generated.len() >= max_new_tokens {
            break;
        }
        logits = model.forward(&[token], cache, true)?;
        logits.eval()?;
        seen.push(token);
    }
    Ok(())
}

/// What the speculative loop needs besides its state.
struct SpecPlan<'a> {
    model: &'a Inkling2Model,
    /// The layer visits whose stream the draft head reads.
    tap_ids: &'a [usize],
    /// Rows of taps the ring keeps.
    keep: usize,
    eos: &'a [u32],
    max_new_tokens: usize,
    /// Most drafts checked per block.
    depth: usize,
    /// Let a [`DepthController`] pick the depth per block, up to `depth`; off, every block checks
    /// `depth` drafts (the tests that pin the loop's bookkeeping).
    adaptive: bool,
}

/// What a decode step and a block cost, as ratios (Inkling 2 Mini on an M3 Max, measured with
/// `examples/inkling2_spec.rs`): a plain step 41 ms; a block of `k` drafts costs the draft (9 ms)
/// and a pass over `k + 1` tokens (49 + 6.1 k ms). With the fused regions the plain step is 33 ms
/// and a pass over `k + 1` tokens 41.8 + 5.7 k ms (`inkling2_profile --block`); the constants were
/// left at the older ratios on purpose: the measured ones made the controller try blocks on text
/// the draft cannot predict (a poem and a translation ran 2% under the plain loop), the older ones
/// kept those at the plain speed and the predictable text at 58 to 59 tok/s.
const STEP_MS: f64 = 41.0;
const BLOCK_BASE_MS: f64 = 58.0;
const BLOCK_PER_DRAFT_MS: f64 = 6.1;
/// A draft's chance of agreeing is estimated as a running mean with this weight on the newest block.
const AGREE_RATE: f64 = 0.15;
/// Before anything is observed a draft is assumed to agree this often (given the ones before it did).
const AGREE_PRIOR: f64 = 0.6;
/// When plain steps look best, a block of one draft is tried every this many steps to see whether
/// the text became predictable again.
const PROBE_EVERY: usize = 6;

/// Picks how many drafts to check per block from how often recent drafts agreed, so text the draft
/// cannot predict does not run slower than the plain loop: it keeps, for each position of the
/// block, the chance that the draft agrees given the ones before it did, and takes the depth with
/// the most tokens per millisecond (depth 0 is a plain step).
struct DepthController {
    agree: Vec<f64>,
    plain_steps: usize,
}

impl DepthController {
    fn new(max: usize) -> Self {
        Self {
            agree: vec![AGREE_PRIOR; max],
            plain_steps: 0,
        }
    }

    fn choose(&mut self) -> usize {
        let (mut best, mut best_rate) = (0, 1.0 / STEP_MS);
        let (mut reach, mut tokens) = (1.0, 1.0);
        for (i, a) in self.agree.iter().enumerate() {
            reach *= a;
            tokens += reach;
            let k = i + 1;
            let rate = tokens / (BLOCK_BASE_MS + BLOCK_PER_DRAFT_MS * k as f64);
            if rate > best_rate {
                (best, best_rate) = (k, rate);
            }
        }
        if best > 0 {
            self.plain_steps = 0;
            return best;
        }
        self.plain_steps += 1;
        if self.plain_steps >= PROBE_EVERY && !self.agree.is_empty() {
            self.plain_steps = 0;
            return 1;
        }
        0
    }

    /// A block checked `k` drafts and the first `agreed` of them agreed: every draft up to the
    /// first miss was a trial of its position.
    fn observe(&mut self, k: usize, agreed: usize) {
        for i in 0..k.min(agreed + 1) {
            let hit = if i < agreed { 1.0 } else { 0.0 };
            self.agree[i] += AGREE_RATE * (hit - self.agree[i]);
        }
    }
}

/// The greedy decode loop with a drafter: each cycle `propose` gives the block after the newest
/// token, the model checks `[that token, drafts...]` in one pass, and the tokens it would itself
/// have chosen are the ones kept (the leading drafts it agrees with, and its own choice after
/// them). The positions after the last kept one are rolled back. Ends where the plain loop does:
/// at the end-of-turn marker (fed), a stop from `on_token`, or the limit.
///
/// `propose(anchor, rows, want, generated)` returns `want` drafts (or more) for the block after
/// `anchor`, the newest token, with `rows` the taps of the positions fed since the last call and
/// `generated` the number of tokens so far; with `want == 0` it only takes in `rows`.
#[allow(clippy::too_many_arguments)]
fn run_speculative(
    plan: &SpecPlan<'_>,
    cache: &mut Inkling2Cache,
    ring: &mut TapRing,
    seen: &mut Vec<u32>,
    generated: &mut Vec<u32>,
    first_logits: &Array,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
    stats: &mut SpecStats,
    sampling: Option<&SamplingConfig>,
    rng: &mut Xorshift64,
    propose: &mut dyn FnMut(u32, Option<&Array>, usize, usize) -> Result<Vec<u32>>,
) -> Result<()> {
    let mut pending: Option<Array> = None;
    let mut controller = DepthController::new(plan.depth);

    // Whether a token ends the turn: the marker, a stop from the sink, or the limit.
    enum Flow {
        More,
        Done { marker: bool },
    }
    let mut emit = |token: u32, generated: &mut Vec<u32>| -> Result<Flow> {
        generated.push(token);
        let stop = on_token(token)?;
        Ok(if plan.eos.contains(&token) {
            Flow::Done { marker: true }
        } else if stop || generated.len() >= plan.max_new_tokens {
            Flow::Done { marker: false }
        } else {
            Flow::More
        })
    };

    let mut anchor = pick(first_logits, seen, sampling, rng)?;
    let mut flow = emit(anchor, generated)?;
    while matches!(flow, Flow::More) {
        let remaining = plan.max_new_tokens - generated.len();
        let want = if plan.adaptive {
            controller.choose()
        } else {
            plan.depth
        };
        let k = want.min(remaining.saturating_sub(1));

        let drafted_at = std::time::Instant::now();
        let block = propose(anchor, pending.take().as_ref(), k, generated.len())?;
        ensure!(
            block.len() >= k,
            "the drafter gave {} tokens for {k}",
            block.len()
        );
        stats.draft_ms += drafted_at.elapsed().as_secs_f64() * 1e3;

        let verified_at = std::time::Instant::now();
        let mut ids = Vec::with_capacity(k + 1);
        ids.push(anchor);
        ids.extend_from_slice(&block[..k]);
        let (logits, taps) = plan.model.verify(&ids, cache, plan.tap_ids)?;
        // `chosen` are the tokens this block commits, in order: the drafts that were accepted and
        // then the model's own token (a correction after a rejected draft, or the next one).
        let chosen: Vec<u32> = match sampling {
            None => {
                let choice = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false)?;
                mlx_rs::transforms::eval([&choice, &taps])?;
                let all: Vec<u32> = choice
                    .as_dtype(mlx_rs::Dtype::Uint32)?
                    .as_slice::<u32>()
                    .to_vec();
                let agreed = (0..k).take_while(|&i| block[i] == all[i]).count();
                all[..=agreed].to_vec()
            }
            Some(cfg) => {
                let flat = mlx_rs::ops::reshape(&logits, &[-1])?;
                mlx_rs::transforms::eval([&flat, &taps])?;
                let vocab = plan.model.args().vocab_size;
                sampled_block(
                    flat.as_slice::<f32>(),
                    vocab,
                    &block[..k],
                    seen,
                    anchor,
                    cfg,
                    rng,
                )
            }
        };
        stats.verify_ms += verified_at.elapsed().as_secs_f64() * 1e3;

        let agreed = chosen.len() - 1;
        stats.cycles += 1;
        stats.drafted += k;
        stats.accepted += agreed;
        if plan.adaptive {
            controller.observe(k, agreed);
        }

        // Emit the model's own tokens in order, until one ends the turn.
        let mut kept = 0;
        for &token in &chosen {
            kept += 1;
            flow = emit(token, generated)?;
            if !matches!(flow, Flow::More) {
                break;
            }
        }
        // The pass fed `[anchor, drafts...]`; what stays is the anchor and the kept tokens but the
        // last, which is the next anchor and not fed yet.
        let rejected = ids.len() - kept;
        if rejected > 0 {
            cache.rollback(rejected)?;
        }
        let rows = taps.try_index((.., ..kept as i32, ..))?;
        ring.push(&rows, plan.keep)?;
        pending = Some(rows);
        seen.push(anchor);
        seen.extend_from_slice(&chosen[..kept - 1]);
        anchor = chosen[kept - 1];
    }
    if let Flow::Done { marker: true } = flow {
        // The end of the turn is part of what a reply read back will contain.
        let taps = plan
            .model
            .advance_taps(&[anchor], cache, plan.tap_ids, None)?;
        ring.push(&taps, plan.keep)?;
        seen.push(anchor);
    }
    Ok(())
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

/// How many tokens a reply may run: what the client asked for (`0` is the server's "until the model
/// ends its turn"), and never more than the context has room for after the prompt. A reply that
/// fills the context ends there as a length cut, not past `max_context` where the caches grow
/// without a bound.
fn reply_budget(max_context: usize, prompt_len: usize, asked: usize) -> usize {
    let room = max_context.saturating_sub(prompt_len);
    if asked == 0 { room } else { asked.min(room) }
}

/// `turns` with each run of tool results in the order of the calls they answer, and `images` (the
/// encoded images of each turn) moved along with them. The template pairs a result with a call by
/// position, so results a client sent in the order they finished would be read as answers to the
/// wrong calls. Borrowed when nothing moves.
fn in_call_order<'t, 'a>(
    turns: &'t [ChatTurn<'a>],
    images: &'t [Vec<Vec<u8>>],
) -> (
    std::borrow::Cow<'t, [ChatTurn<'a>]>,
    std::borrow::Cow<'t, [Vec<Vec<u8>>]>,
) {
    use std::borrow::Cow;
    let Some(order) = tool_result_order(turns) else {
        return (Cow::Borrowed(turns), Cow::Borrowed(images));
    };
    let moved = order.iter().map(|&i| turns[i].clone()).collect();
    let images = if images.is_empty() {
        Cow::Borrowed(images)
    } else {
        Cow::Owned(
            order
                .iter()
                .map(|&i| images.get(i).cloned().unwrap_or_default())
                .collect(),
        )
    };
    (Cow::Owned(moved), images)
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

    /// RED before: a `max_tokens` of 0 (the server's "until the model ends its turn") gave an empty
    /// reply, and a large one was bounded by nothing but the memory the caches could grow into.
    #[test]
    fn a_reply_is_bounded_by_what_the_context_has_room_for_and_zero_is_that_room() {
        assert_eq!(reply_budget(1000, 100, 50), 50);
        assert_eq!(reply_budget(1000, 100, 0), 900);
        assert_eq!(reply_budget(1000, 100, 200_000), 900);
        assert_eq!(reply_budget(1000, 999, 0), 1);
        assert_eq!(reply_budget(1000, 1000, 0), 0);
    }

    /// RED before: tool results sent in the order they finished were read as the answers to other
    /// calls, and what a tool turn carries (its images) has to move with it.
    #[test]
    fn tool_results_follow_their_calls_and_take_their_images_along() {
        let args = serde_json::json!({});
        let calls = [
            crate::chat_io::AssistantToolCall {
                id: "a",
                name: "f",
                arguments: &args,
            },
            crate::chat_io::AssistantToolCall {
                id: "b",
                name: "f",
                arguments: &args,
            },
        ];
        let result = |id| ChatTurn::Tool {
            tool_call_id: id,
            name: None,
            content: id,
        };
        let turns = [
            ChatTurn::User("q"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: &calls,
            },
            result("b"),
            result("a"),
        ];
        let images = vec![vec![], vec![], vec![vec![2u8]], vec![vec![1u8]]];
        let (ordered, moved) = in_call_order(&turns, &images);
        let contents: Vec<&str> = ordered
            .iter()
            .filter_map(|t| match t {
                ChatTurn::Tool { content, .. } => Some(*content),
                _ => None,
            })
            .collect();
        assert_eq!(contents, ["a", "b"]);
        assert_eq!(
            moved[2],
            vec![vec![1u8]],
            "a's picture sits with a's result"
        );
        assert_eq!(moved[3], vec![vec![2u8]]);
        // In order already: nothing is copied.
        let (same, same_images) = in_call_order(&turns[..2], &images[..2]);
        assert!(matches!(same, std::borrow::Cow::Borrowed(_)));
        assert!(matches!(same_images, std::borrow::Cow::Borrowed(_)));
    }

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
    fn image_runs_are_the_maximal_runs_of_the_image_token() {
        const IMG: u32 = 9;
        assert_eq!(image_runs(&[], IMG), vec![]);
        assert_eq!(image_runs(&[1, 2, 3], IMG), vec![]);
        assert_eq!(image_runs(&[IMG, IMG, 1], IMG), vec![(0, 2)]);
        assert_eq!(
            image_runs(&[1, IMG, IMG, IMG, 2, IMG, 3, IMG, IMG], IMG),
            vec![(1, 3), (5, 1), (7, 2)]
        );
    }

    #[test]
    fn two_images_of_one_grid_are_different_prompts_and_the_same_image_is_not() {
        // The prompts are token for token alike; what tells them apart is the hash of the image.
        let prompt = |picture: &[u8]| -> Vec<Key> {
            let hash = image_hash(picture);
            vec![(1, 0), (9, hash), (9, hash), (2, 0), (3, 0)]
        };
        let (red, red_again, green) = (prompt(b"red"), prompt(b"red"), prompt(b"green"));
        assert_eq!(red, red_again);
        assert_ne!(red, green);
        // A retained conversation that saw the red picture can be taken back to the text before
        // the image (position 1) for the green one, but not past it.
        assert_eq!(reusable_point([1, 4].into_iter(), &red, &green), Some(1));
        // And for the red one again, to the longest point shorter than the new prompt.
        assert_eq!(
            reusable_point([1, 4].into_iter(), &red, &red_again),
            Some(4)
        );
    }

    #[test]
    fn the_depth_controller_checks_many_drafts_when_they_agree_and_none_when_they_do_not() {
        // Untried, a draft is assumed to agree often enough to be worth a block.
        let mut fresh = DepthController::new(7);
        let first = fresh.choose();
        assert!((2..=4).contains(&first), "{first}");

        // Every block agreeing all the way: the deepest block.
        let mut sure = DepthController::new(7);
        for _ in 0..40 {
            sure.observe(7, 7);
        }
        assert_eq!(sure.choose(), 7);

        // Never agreeing: plain steps, with a block of one every PROBE_EVERY steps to look again.
        let mut lost = DepthController::new(7);
        for _ in 0..40 {
            lost.observe(3, 0);
        }
        let picks: Vec<usize> = (0..2 * PROBE_EVERY).map(|_| lost.choose()).collect();
        assert_eq!(picks.iter().filter(|&&k| k == 1).count(), 2, "{picks:?}");
        assert!(picks.iter().all(|&k| k <= 1), "{picks:?}");
        assert_eq!(picks[PROBE_EVERY - 1], 1, "{picks:?}");

        // Text that becomes predictable again is noticed through the probes.
        for _ in 0..30 {
            lost.observe(1, 1);
        }
        assert!(lost.choose() >= 2);
    }

    #[test]
    fn only_the_drafts_up_to_the_first_miss_count_as_trials() {
        let mut c = DepthController::new(7);
        let before = c.agree.clone();
        // 7 checked, the first 2 agreed: positions 1 and 2 hit, position 3 missed, 4 to 7 untried.
        c.observe(7, 2);
        assert!(c.agree[0] > before[0] && c.agree[1] > before[1]);
        assert!(c.agree[2] < before[2]);
        assert_eq!(&c.agree[3..], &before[3..]);
        // Nothing checked, nothing learned.
        let mut d = DepthController::new(7);
        d.observe(0, 0);
        assert_eq!(d.agree, before);
    }

    fn cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 64,
            repeat_penalty: 1.0,
            repeat_penalty_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            seed: 5,
            dry: lumen_core::dry::dry_config_from_env(),
        }
    }

    /// The sampled block rule against plain sampling on a small made-up model: the next token's
    /// logits depend on the token before it (a fixed table), a drafter that is right some of the
    /// time proposes blocks of two, and the first three tokens of many speculative runs have the
    /// distribution the product of the plain steps gives.
    #[test]
    fn the_sampled_block_rule_draws_what_plain_sampling_draws() {
        const V: usize = 5;
        // Logits of the next token given the one before; a table with some structure and no ties.
        let table: [[f32; V]; V] = [
            [0.2, 1.4, -0.3, 0.9, 0.0],
            [1.1, -0.5, 0.7, 0.3, 1.6],
            [-0.2, 0.8, 1.3, -0.9, 0.5],
            [0.6, 0.1, -0.4, 1.2, 0.9],
            [1.0, 0.4, 0.2, -0.6, 0.8],
        ];
        // A drafter that disagrees with the model: its own table.
        let draft_table: [[f32; V]; V] = [
            [0.0, 0.5, 1.2, 0.1, 0.3],
            [0.9, 0.2, 0.4, 1.5, 0.0],
            [0.3, 1.1, 0.0, 0.2, 0.6],
            [1.4, 0.0, 0.3, 0.8, 0.1],
            [0.2, 0.7, 0.9, 0.0, 1.3],
        ];
        let argmax = |row: &[f32; V]| -> u32 {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32
        };
        let c = SamplingConfig {
            temperature: 0.9,
            top_k: 0,
            ..cfg()
        };
        let probs = |prev: usize| -> Vec<f32> {
            let mut row = table[prev].to_vec();
            sampling_distribution(&mut row, &[], &c)
        };
        let (start, steps, trials) = (2usize, 3usize, 120_000usize);
        // What plain sampling gives: the product of the steps.
        let mut expected = vec![0.0f64; V.pow(steps as u32)];
        for a in 0..V {
            for b in 0..V {
                for d in 0..V {
                    expected[(a * V + b) * V + d] = f64::from(probs(start)[a])
                        * f64::from(probs(a)[b])
                        * f64::from(probs(b)[d]);
                }
            }
        }
        let mut rng = Xorshift64::new(99);
        let mut counts = vec![0usize; expected.len()];
        for _ in 0..trials {
            let mut tokens: Vec<u32> = Vec::new();
            let mut prev = start;
            while tokens.len() < steps {
                // A block of two drafts from the drafter's own table.
                let d1 = argmax(&draft_table[prev]);
                let d2 = argmax(&draft_table[d1 as usize]);
                let mut rows = Vec::new();
                for p in [prev, d1 as usize, d2 as usize] {
                    rows.extend_from_slice(&table[p]);
                }
                let committed = sampled_block(&rows, V, &[d1, d2], &[], prev as u32, &c, &mut rng);
                for t in committed {
                    if tokens.len() < steps {
                        tokens.push(t);
                    }
                }
                prev = *tokens.last().unwrap() as usize;
            }
            counts[(tokens[0] as usize * V + tokens[1] as usize) * V + tokens[2] as usize] += 1;
        }
        let tv: f64 = counts
            .iter()
            .zip(&expected)
            .map(|(&n, &e)| (n as f64 / trials as f64 - e).abs())
            .sum::<f64>()
            / 2.0;
        eprintln!("total variation from plain sampling over {steps} tokens: {tv:.4}");
        assert!(
            tv < 0.02,
            "total variation {tv:.4} from plain sampling over {steps} tokens"
        );
        // The drafter really is wrong a good part of the time, so the rule was exercised.
        let mut rejected = 0;
        for _ in 0..2_000 {
            let d1 = argmax(&draft_table[start]);
            let mut rows = table[start].to_vec();
            rows.extend_from_slice(&table[d1 as usize]);
            let committed = sampled_block(&rows, V, &[d1], &[], start as u32, &c, &mut rng);
            rejected += usize::from(committed.len() == 1);
        }
        assert!(
            rejected > 300 && rejected < 1_900,
            "{rejected} of 2000 rejected"
        );
    }

    /// The anchor is in the penalty window of the first position and every accepted draft is in the
    /// window of the next: a strong repeat penalty on a token must take its probability down there.
    #[test]
    fn the_penalty_window_holds_the_anchor_and_the_accepted_drafts() {
        const V: usize = 6;
        let c = SamplingConfig {
            repeat_penalty: 8.0,
            top_k: 0,
            ..cfg()
        };
        let row: Vec<f32> = vec![0.5, 3.0, 2.8, 2.5, 0.1, 1.0];
        let mut rows = row.clone();
        rows.extend_from_slice(&row);
        let plain = |recent: &[u32]| -> Vec<f32> {
            let mut r = row.clone();
            sampling_distribution(&mut r, recent, &c)
        };
        let mut rng = Xorshift64::new(17);
        // Position 0 after anchor 1: token 1 is penalized there.
        let (with, without) = (plain(&[1]), plain(&[]));
        assert!(
            with[1] < without[1] - 0.2,
            "the penalty does not bite: {} against {}",
            with[1],
            without[1]
        );
        let n = 30_000;
        let hits = (0..n)
            .filter(|_| sampled_block(&rows, V, &[], &[], 1, &c, &mut rng)[0] == 1)
            .count();
        assert!(
            (hits as f32 / n as f32 - with[1]).abs() < 0.02,
            "{hits} of {n} against {}",
            with[1]
        );
        // Position 1 after the anchor 1 and the accepted draft 3: both are penalized there.
        let after = plain(&[1, 3]);
        let (mut second, mut seconds) = (0usize, 0usize);
        for _ in 0..40_000 {
            let committed = sampled_block(&rows, V, &[3], &[], 1, &c, &mut rng);
            if committed.len() == 2 {
                seconds += 1;
                second += usize::from(committed[1] == 3);
            }
        }
        assert!(seconds > 1_000, "{seconds}");
        assert!(
            (second as f32 / seconds as f32 - after[3]).abs() < 0.03,
            "{second} of {seconds} against {}",
            after[3]
        );
    }

    #[test]
    fn a_draft_the_sampling_pipeline_forbids_is_never_accepted() {
        const V: usize = 8;
        let c = SamplingConfig { top_k: 2, ..cfg() };
        // Token 0 is far below the top two, so top-k leaves it no mass.
        let row: Vec<f32> = vec![-9.0, 5.0, 4.0, 1.0, 0.0, 0.5, 0.2, 0.1];
        let mut rows = row.clone();
        rows.extend_from_slice(&row);
        let mut rng = Xorshift64::new(3);
        for _ in 0..500 {
            let committed = sampled_block(&rows, V, &[0], &[], 1, &c, &mut rng);
            assert_eq!(committed.len(), 1, "a forbidden draft was accepted");
            assert!(committed[0] == 1 || committed[0] == 2, "{committed:?}");
        }
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

/// The backend on the real checkpoint. Set `LUMEN_INKLING2_MODEL_DIR` to the checkpoint directory:
///
/// ```text
/// LUMEN_INKLING2_MODEL_DIR=~/.slotstream/models/Inkling-2-Mini-oQ4e-mtp \
///   cargo test -p lumen-mlx --features mlx-native --lib inkling2::backend -- --ignored --test-threads=1
/// ```
/// The speculative loop against the plain one on the tiny replica, with a drafter that is told the
/// plain continuation and corrupts it where a case wants: the kept tokens, the rollback, the end of
/// the turn and the limits must come out as the plain loop's, whatever the drafter says.
#[cfg(test)]
mod speculative_tests {
    use super::*;
    use crate::inkling2::tests::{fixture_dir, load_replica};

    const TAPS: [usize; 4] = [5, 23, 41, 48];
    const KEEP: usize = 11;

    struct Outcome {
        generated: Vec<u32>,
        seen: Vec<u32>,
        cache_len: usize,
        ring_end: usize,
        stats: SpecStats,
    }

    fn prompt() -> Vec<u32> {
        let want =
            Array::load_safetensors(fixture_dir().join("reference").join("outputs.safetensors"))
                .expect("reference");
        want["ids"].as_slice::<i32>()[..30]
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    /// The prompt fed, as the backend feeds it: all but the last token, then the last one for logits.
    fn prefilled(model: &Inkling2Model, prompt: &[u32]) -> (Inkling2Cache, TapRing, Array) {
        let mut cache = model.make_cache();
        let mut ring = TapRing::new();
        let last = prompt.len() - 1;
        let taps = model
            .advance_taps(&prompt[..last], &mut cache, &TAPS, None)
            .unwrap();
        ring.push(&taps, KEEP).unwrap();
        let (logits, taps) = model
            .forward_taps(&prompt[last..], &mut cache, true, &TAPS, None, false)
            .unwrap();
        logits.eval().unwrap();
        taps.eval().unwrap();
        ring.push(&taps, KEEP).unwrap();
        (cache, ring, logits)
    }

    /// Runs the plain loop and returns its outcome, stopping after `stop_after` tokens when given.
    fn run_plain_case(
        model: &Inkling2Model,
        eos: &[u32],
        prompt: &[u32],
        max_new: usize,
        stop_after: Option<usize>,
    ) -> Outcome {
        let (mut cache, _ring, logits) = prefilled(model, prompt);
        let (mut seen, mut generated) = (prompt.to_vec(), Vec::new());
        let mut calls = 0usize;
        run_plain(
            model,
            eos,
            &mut cache,
            &mut seen,
            &mut generated,
            logits,
            max_new,
            None,
            &mut Xorshift64::new(1),
            &mut |_| {
                calls += 1;
                Ok(stop_after == Some(calls))
            },
        )
        .unwrap();
        Outcome {
            cache_len: cache.len(),
            ring_end: cache.len(),
            generated,
            seen,
            stats: SpecStats::default(),
        }
    }

    /// [`run_spec_case_with`] with the depth fixed: what pins the loop's bookkeeping.
    #[allow(clippy::too_many_arguments)]
    fn run_spec_case(
        model: &Inkling2Model,
        eos: &[u32],
        prompt: &[u32],
        max_new: usize,
        stop_after: Option<usize>,
        depth: usize,
        oracle: &[u32],
        wrong: Option<usize>,
        all_wrong: bool,
    ) -> Outcome {
        run_spec_case_with(
            model, eos, prompt, max_new, stop_after, depth, oracle, wrong, all_wrong, false, None,
        )
    }

    /// The speculative loop with a drafter that proposes `oracle[at..at + want]`, with the entry at
    /// block position `wrong` (if any) replaced by a token that cannot be right.
    #[allow(clippy::too_many_arguments)]
    fn run_spec_case_with(
        model: &Inkling2Model,
        eos: &[u32],
        prompt: &[u32],
        max_new: usize,
        stop_after: Option<usize>,
        depth: usize,
        oracle: &[u32],
        wrong: Option<usize>,
        all_wrong: bool,
        adaptive: bool,
        sampling: Option<&SamplingConfig>,
    ) -> Outcome {
        let (mut cache, mut ring, logits) = prefilled(model, prompt);
        let (mut seen, mut generated) = (prompt.to_vec(), Vec::new());
        let mut stats = SpecStats::default();
        let plan = SpecPlan {
            model,
            tap_ids: &TAPS,
            keep: KEEP,
            eos,
            max_new_tokens: max_new,
            depth,
            adaptive,
        };
        let mut calls = 0usize;
        let mut taken = 0usize;
        run_speculative(
            &plan,
            &mut cache,
            &mut ring,
            &mut seen,
            &mut generated,
            &logits,
            &mut |_| {
                calls += 1;
                Ok(stop_after == Some(calls))
            },
            &mut stats,
            sampling,
            &mut Xorshift64::new(11),
            &mut |_anchor, rows, want, at| {
                if let Some(rows) = rows {
                    taken += rows.shape()[1] as usize;
                }
                let mut block: Vec<u32> = (0..want.max(1))
                    .map(|i| oracle.get(at + i).copied().unwrap_or(0))
                    .collect();
                for (i, t) in block.iter_mut().enumerate() {
                    if all_wrong || wrong == Some(i) {
                        *t = (*t + 1) % 250;
                    }
                }
                Ok(block)
            },
        )
        .unwrap();
        // From the second cycle on, the drafter is handed the taps of the positions just fed.
        assert!(
            stats.cycles <= 1 || taken > 0,
            "the drafter was never handed the taps of a fed position"
        );
        Outcome {
            cache_len: cache.len(),
            ring_end: ring.end(),
            generated,
            seen,
            stats,
        }
    }

    fn same(label: &str, got: &Outcome, want: &Outcome) {
        assert_eq!(got.generated, want.generated, "{label}: the tokens");
        assert_eq!(got.seen, want.seen, "{label}: the tokens fed");
        assert_eq!(got.cache_len, want.cache_len, "{label}: the cache length");
        assert_eq!(
            got.ring_end, got.cache_len,
            "{label}: the taps end where the cache does"
        );
    }

    #[test]
    fn a_drafter_that_is_right_wrong_or_half_right_changes_nothing_but_speed() {
        let model = load_replica();
        let prompt = prompt();
        let free = run_plain_case(&model, &[], &prompt, 30, None);
        for depth in [3usize, 7] {
            // Right: every draft is kept and a cycle gives depth + 1 tokens.
            let right = run_spec_case(
                &model,
                &[],
                &prompt,
                30,
                None,
                depth,
                &free.generated,
                None,
                false,
            );
            same(&format!("right, depth {depth}"), &right, &free);
            assert_eq!(right.stats.accepted, right.stats.drafted, "depth {depth}");
            assert!(
                right.stats.tokens_per_cycle() > depth as f64 * 0.5,
                "depth {depth}"
            );
            // Wrong at one position of the block: the tokens before it are kept, the rest rolled back.
            for at in [0usize, 2, 6] {
                let case = run_spec_case(
                    &model,
                    &[],
                    &prompt,
                    30,
                    None,
                    depth,
                    &free.generated,
                    Some(at),
                    false,
                );
                same(&format!("wrong at {at}, depth {depth}"), &case, &free);
            }
            // Wrong everywhere: one token a cycle, every draft rolled back.
            let wrong = run_spec_case(
                &model,
                &[],
                &prompt,
                30,
                None,
                depth,
                &free.generated,
                None,
                true,
            );
            same(&format!("all wrong, depth {depth}"), &wrong, &free);
            assert_eq!(wrong.stats.accepted, 0);
            assert_eq!(
                wrong.stats.cycles, 29,
                "the first token comes from the prefill"
            );
        }
    }

    #[test]
    fn the_limit_the_marker_and_a_stop_end_the_turn_where_the_plain_loop_ends_it() {
        let model = load_replica();
        let prompt = prompt();
        let free = run_plain_case(&model, &[], &prompt, 30, None);
        // The limit, inside a block.
        for limit in [1usize, 5, 8] {
            let want = run_plain_case(&model, &[], &prompt, limit, None);
            let got = run_spec_case(
                &model,
                &[],
                &prompt,
                limit,
                None,
                7,
                &free.generated,
                None,
                false,
            );
            same(&format!("limit {limit}"), &got, &want);
            assert_eq!(got.generated.len(), limit);
        }
        // The marker in the middle of an accepted block: it is the last token and it is fed.
        for marker_at in [3usize, 10] {
            let eos = [free.generated[marker_at]];
            let want = run_plain_case(&model, &eos, &prompt, 30, None);
            assert_eq!(
                want.generated.len(),
                marker_at + 1,
                "the plain loop stops at the marker"
            );
            {
                let depth = 7usize;
                let got = run_spec_case(
                    &model,
                    &eos,
                    &prompt,
                    30,
                    None,
                    depth,
                    &free.generated,
                    None,
                    false,
                );
                same(
                    &format!("marker at {marker_at}, depth {depth}"),
                    &got,
                    &want,
                );
            }
        }
        // A stop from the sink after the 7th token: not fed, as in the plain loop.
        let want = run_plain_case(&model, &[], &prompt, 30, Some(7));
        {
            let depth = 3usize;
            let got = run_spec_case(
                &model,
                &[],
                &prompt,
                30,
                Some(7),
                depth,
                &free.generated,
                None,
                false,
            );
            same(&format!("stop at 7, depth {depth}"), &got, &want);
            assert_eq!(got.generated.len(), 7);
        }
    }

    #[test]
    fn an_adaptive_loop_stops_drafting_when_the_drafts_are_wrong_and_changes_nothing_else() {
        let model = load_replica();
        let prompt = prompt();
        let free = run_plain_case(&model, &[], &prompt, 30, None);
        let fixed = run_spec_case(
            &model,
            &[],
            &prompt,
            30,
            None,
            7,
            &free.generated,
            None,
            true,
        );
        let adaptive = run_spec_case_with(
            &model,
            &[],
            &prompt,
            30,
            None,
            7,
            &free.generated,
            None,
            true,
            true,
            None,
        );
        same("adaptive, all wrong", &adaptive, &free);
        assert_eq!(adaptive.stats.accepted, 0);
        // The same tokens, with far fewer drafts put to the check: the controller went to plain steps.
        assert!(
            adaptive.stats.drafted * 2 < fixed.stats.drafted,
            "{} drafts checked against {}",
            adaptive.stats.drafted,
            fixed.stats.drafted
        );
        // And when the drafts are right it keeps asking for a full block.
        let right = run_spec_case_with(
            &model,
            &[],
            &prompt,
            30,
            None,
            7,
            &free.generated,
            None,
            false,
            true,
            None,
        );
        same("adaptive, right", &right, &free);
        assert_eq!(right.stats.accepted, right.stats.drafted);
        assert!(
            right.stats.tokens_per_cycle() > 4.0,
            "{}",
            right.stats.tokens_per_cycle()
        );
    }

    fn cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 64,
            repeat_penalty: 1.0,
            repeat_penalty_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            seed: 5,
            dry: lumen_core::dry::dry_config_from_env(),
        }
    }

    /// A sampled request on the tiny replica: the bookkeeping is the plain loop's whatever the
    /// drafter says and whatever the draws are, and one seed gives one reply.
    #[test]
    fn a_sampled_speculative_loop_keeps_the_books_and_is_reproducible() {
        let model = load_replica();
        let prompt = prompt();
        let free = run_plain_case(&model, &[], &prompt, 30, None);
        // Nearly greedy, so that the sampled continuation follows the plain one the oracle proposes
        // and its drafts are accepted a fair share of the time; the draws still go through the
        // sampled rule.
        let c = SamplingConfig {
            temperature: 0.02,
            top_k: 8,
            ..cfg()
        };
        for wrong in [None, Some(1usize)] {
            let a = run_spec_case_with(
                &model,
                &[],
                &prompt,
                30,
                None,
                7,
                &free.generated,
                wrong,
                false,
                false,
                Some(&c),
            );
            assert_eq!(a.generated.len(), 30);
            assert_eq!(
                a.cache_len,
                prompt.len() + 29,
                "every token but the last is fed"
            );
            assert_eq!(a.ring_end, a.cache_len);
            assert_eq!(a.seen.len(), a.cache_len);
            assert_eq!(&a.seen[prompt.len()..], &a.generated[..29]);
            assert!(
                a.generated.iter().all(|&t| t < 250),
                "a token past the live vocabulary"
            );
            let b = run_spec_case_with(
                &model,
                &[],
                &prompt,
                30,
                None,
                7,
                &free.generated,
                wrong,
                false,
                false,
                Some(&c),
            );
            assert_eq!(a.generated, b.generated, "the same seed, the same reply");
            assert!(
                a.stats.accepted > 0 && a.stats.accepted < a.stats.drafted,
                "{:?}",
                a.stats
            );
        }
        // The marker and a stop end the turn as they do when drafts are checked by argmax.
        let eos = [free.generated[3]];
        let sampled = run_spec_case_with(
            &model,
            &eos,
            &prompt,
            30,
            None,
            7,
            &free.generated,
            None,
            false,
            false,
            Some(&c),
        );
        if let Some(at) = sampled.generated.iter().position(|t| *t == eos[0]) {
            assert_eq!(sampled.generated.len(), at + 1, "nothing after the marker");
            assert_eq!(
                sampled.cache_len,
                prompt.len() + at + 1,
                "the marker is fed"
            );
        }
    }

    #[test]
    fn the_drafter_is_asked_for_nothing_when_a_cycle_has_one_token_left() {
        let model = load_replica();
        let prompt = prompt();
        let free = run_plain_case(&model, &[], &prompt, 30, None);
        // With 2 tokens allowed, the first comes from the prefill and the second from one cycle
        // that checks no draft at all (its one token is the limit).
        let got = run_spec_case(
            &model,
            &[],
            &prompt,
            2,
            None,
            7,
            &free.generated,
            None,
            false,
        );
        assert_eq!((got.stats.cycles, got.stats.drafted), (1, 0));
        assert_eq!(got.generated.len(), 2);
    }
}

#[cfg(test)]
mod real_checkpoint_tests {
    use super::*;

    fn backend() -> Inkling2Backend {
        let dir = std::env::var("LUMEN_INKLING2_MODEL_DIR").expect("LUMEN_INKLING2_MODEL_DIR");
        Inkling2Backend::from_dir("inkling2", Path::new(&dir)).expect("load")
    }

    /// Prompts of the regime comparisons: short answers, code, arithmetic, translation, lists.
    const PROMPTS: [&str; 20] = [
        "Name one primary color. One word.",
        "What is the capital of France? One short sentence.",
        "Write a Python function that returns the nth Fibonacci number, with a docstring.",
        "Explain in three sentences why the sky is blue.",
        "Translate to French: 'The meeting has been moved to Thursday afternoon.'",
        "Write a haiku about autumn rain.",
        "What is 17 * 23? Show the steps.",
        "List five prime numbers greater than 50.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Write a SQL query that finds the second highest salary in an employees table.",
        "Give me a short packing list for a three day hiking trip.",
        "What is the difference between a process and a thread?",
        "Write a limerick about a cat who learns to code.",
        "Translate to Korean: 'Where is the nearest train station?'",
        "Is 9973 a prime number? Explain briefly.",
        "Write a bash one-liner that counts the lines of every .rs file in a directory.",
        "Describe the water cycle for a ten year old.",
        "If a train leaves at 3:15 pm and the trip takes 2 hours 50 minutes, when does it arrive?",
        "Give two pros and two cons of using a monorepo.",
        "Write a short, polite email declining a meeting invitation.",
    ];

    /// One greedy turn over `turns`, returning what the model said.
    fn greedy(b: &mut Inkling2Backend, turns: &[ChatTurn<'_>], max_new: usize) -> ParsedResponse {
        let ov = SamplingOverrides::default();
        b.chat(
            turns,
            &[],
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
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
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
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
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
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn an_edited_history_falls_back_to_a_cold_prefill() {
        let mut b = backend();
        greedy(&mut b, &[ChatTurn::User("Say hello.")], 16);
        let edited = [ChatTurn::User("Say goodbye.")];
        greedy(&mut b, &edited, 16);
        assert_eq!(b.last_run().reused_tokens, 0);
    }

    /// A reply as a client sends it back: the reasoning inside the envelope, then the answer.
    fn join_envelope(reply: &ParsedResponse) -> String {
        if reply.reasoning.is_empty() {
            reply.visible.clone()
        } else {
            crate::chat_io::join_reasoning_envelope(&reply.reasoning, &reply.visible)
        }
    }

    fn scene(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling2_vision")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// One greedy turn over `turns` with `images`, returning what the model said.
    fn greedy_with_images(
        b: &mut Inkling2Backend,
        turns: &[ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new: usize,
    ) -> ParsedResponse {
        let ov = SamplingOverrides::default();
        b.chat(
            turns,
            images,
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
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn an_image_is_described_and_the_follow_up_reuses_the_caches_without_encoding_it() {
        let mut b = backend();
        let image = scene("scene.png");
        let question = "What shapes are in this picture and what color is each? Be brief.";
        let turns = [ChatTurn::User(question)];
        let images = vec![vec![image.clone()]];

        let first = greedy_with_images(&mut b, &turns, &images, 160);
        let stats = b.last_run();
        println!("first: {:?}\n{stats:?}", first.visible);
        let said = first.visible.to_lowercase();
        assert!(said.contains("red") && said.contains("blue"), "{said}");
        assert!(said.contains("square") && said.contains("circle"), "{said}");
        assert_eq!(stats.images_encoded, 1);
        assert_eq!(stats.reused_tokens, 0);
        // 15 blocks, framed by the role marker, <|content_image|> and <|end_message|>.
        let text_only = b
            .build_prompt(&turns, &[], &ResolvedToolChoice::Auto, false, None)
            .unwrap();
        assert_eq!(
            stats.prompt_tokens,
            text_only.len() + crate::inkling2::image_message_tokens(&image).unwrap(),
            "the count the server adds for an image is the count the prompt has"
        );

        // The reply goes back with the same image and a follow-up: the image is in the caches.
        let reply = join_envelope(&first);
        let follow_up = [
            ChatTurn::User(question),
            ChatTurn::Assistant {
                text: &reply,
                tool_calls: &[],
            },
            ChatTurn::User("And which one is on the left?"),
        ];
        let images = vec![vec![image], vec![], vec![]];
        let second = greedy_with_images(&mut b, &follow_up, &images, 120);
        let stats = b.last_run();
        println!("second: {:?}\n{stats:?}", second.visible);
        assert_eq!(stats.images_encoded, 0, "the image was encoded again");
        // At least the whole first prompt (46 tokens: the picture and the question), and the reply when
        // it reads back as the model wrote it.
        assert!(stats.reused_tokens >= 46, "reused {}", stats.reused_tokens);
        assert!(
            second.visible.to_lowercase().contains("square"),
            "{}",
            second.visible
        );
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn a_different_picture_on_the_same_grid_is_not_taken_for_the_cached_one() {
        let mut b = backend();
        let question = "What is the leftmost shape, and what color? Answer in one short sentence.";
        let turns = [ChatTurn::User(question)];
        let red_first = greedy_with_images(&mut b, &turns, &[vec![scene("scene.png")]], 80);
        assert!(
            red_first.visible.to_lowercase().contains("red"),
            "{}",
            red_first.visible
        );

        // Token for token the same prompt, another picture: nothing past the text before the image
        // may come from the caches, and the answer follows the new picture.
        let yellow = greedy_with_images(&mut b, &turns, &[vec![scene("scene_alt.png")]], 80);
        let stats = b.last_run();
        println!("alt: {:?}\n{stats:?}", yellow.visible);
        assert_eq!(stats.images_encoded, 1);
        assert!(
            stats.reused_tokens < 20,
            "reused {} tokens",
            stats.reused_tokens
        );
        let said = yellow.visible.to_lowercase();
        assert!(said.contains("yellow") || said.contains("circle"), "{said}");
        assert!(!said.contains("red"), "{said}");
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn images_in_the_wrong_message_or_not_an_image_are_refused_as_client_errors() {
        let mut b = backend();
        // A conversation the caches hold: a refused request must leave it there.
        let held = [ChatTurn::User("Name one primary color. One word.")];
        greedy(&mut b, &held, 24);
        let ov = SamplingOverrides::default();
        let mut ask = |turns: &[ChatTurn<'_>], images: &[Vec<Vec<u8>>]| {
            b.chat(
                turns,
                images,
                &[],
                &ResolvedToolChoice::Auto,
                8,
                0.0,
                1.0,
                false,
                &ov,
                &mut |_| Ok(()),
            )
            .expect_err("refused")
        };
        let image = scene("scene.png");
        let in_system = ask(
            &[ChatTurn::System("be brief"), ChatTurn::User("hi")],
            &[vec![image.clone()], vec![]],
        );
        assert!(
            in_system
                .downcast_ref::<crate::chat_io::UnsupportedRequest>()
                .is_some(),
            "{in_system}"
        );
        // A picture whose header reads fine and whose pixels do not: refused after the prompt is
        // built, where the retained conversation could already have been taken.
        let mut cut_short = image.clone();
        cut_short.truncate(cut_short.len() / 2);
        let broken = ask(&[ChatTurn::User("hi")], &[vec![cut_short]]);
        assert!(
            broken
                .downcast_ref::<crate::chat_io::UnsupportedRequest>()
                .is_some(),
            "{broken}"
        );
        let not_an_image = ask(
            &[ChatTurn::User("hi")],
            &[vec![b"definitely not an image".to_vec()]],
        );
        assert!(
            not_an_image
                .downcast_ref::<crate::chat_io::UnsupportedRequest>()
                .is_some(),
            "{not_an_image}"
        );
        greedy(&mut b, &held, 24);
        assert!(
            b.last_run().reused_tokens > 0,
            "the refused requests cost the client its retained conversation"
        );
    }

    /// Speculative decoding keeps the model's own choices, but a pass over a block rounds a little
    /// differently from single-token steps in bfloat16, so it can pick the other of two tokens the
    /// plain loop barely preferred. This feeds the plain loop's own continuation back in blocks of 8
    /// (what a verify pass does) and checks that wherever the block pass disagrees with the plain
    /// choice, the two tokens were close in that pass too.
    ///
    /// Two top logits that are exactly equal (bfloat16 has few values) are a tie, which the argmax
    /// breaks one way or the other, not a disagreement; they are counted apart. The bounds come from
    /// measurement, not from this test's three prompts: over 20 prompts and 2,745 positions per
    /// regime (`a_kernel_flips_the_greedy_choice_only_at_near_ties_like_a_block_pass`) the verify
    /// blocks flip 0.87 to 1.28% of the positions, up to 7.3 bfloat16 steps (the loop's bf16 rounding
    /// staircases turn rounding noise into a flip of a step or two, and now and then of several),
    /// and the default-on kernels 0.7 to 1.1%, up to 8.7. So a flip may be up to 10 steps and may
    /// touch up to 3% of the positions; a wrong verify pass moves most of the logits, not a few.
    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn a_batched_pass_disagrees_with_single_steps_only_at_near_ties() {
        let mut b = backend();
        b.set_draft_depth(None);
        let ov = SamplingOverrides::default();
        // The largest margin in units of the bfloat16 step at the top logit (a bf16 value is only exact to
        // one step, and the two passes can each be a step off).
        let mut worst = 0.0f32;
        let mut disagreements = 0usize;
        let mut ties = 0usize;
        let mut compared = 0usize;
        for (prompt, thinking) in [
            (
                "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring.",
                false,
            ),
            (
                "Explain in three sentences how a hash map handles collisions.",
                false,
            ),
            (
                "A train leaves at 3:15 pm and travels 210 km at 70 km/h. Another leaves the same station at 4:00 pm at 90 km/h on the same track in the same direction. When does the second catch up?",
                true,
            ),
        ] {
            let ids = b
                .build_prompt(
                    &[ChatTurn::User(prompt)],
                    &[],
                    &ResolvedToolChoice::Auto,
                    thinking,
                    None,
                )
                .unwrap();
            let plain = b.generate(&ids, 150, 0.0, 1.0, &ov).unwrap();
            let vocab = b.model.args().vocab_size;
            // The tokens the model is fed from the anchor on, and the plain choice after each.
            let mut fed = vec![*ids.last().unwrap()];
            fed.extend_from_slice(&plain[..plain.len() - 1]);
            // Block sizes a verify pass can have: the loop checks 1 to 8 tokens at a time.
            for size in [2usize, 3, 4, 6, 8] {
                let mut cache = b.model.make_cache();
                b.model.advance(&ids[..ids.len() - 1], &mut cache).unwrap();
                for (n, block) in fed.chunks(size).enumerate() {
                    let (logits, _) = b.model.verify(block, &mut cache, &[5]).unwrap();
                    logits.eval().unwrap();
                    let rows = logits.as_slice::<f32>().to_vec();
                    for (j, row) in rows.chunks(vocab).enumerate() {
                        let want = plain[n * size + j];
                        let got = row
                            .iter()
                            .enumerate()
                            .max_by(|a, c| a.1.total_cmp(c.1))
                            .map(|(i, _)| i as u32)
                            .unwrap();
                        compared += 1;
                        if got != want {
                            let margin = row[got as usize] - row[want as usize];
                            if margin <= 0.0 {
                                ties += 1;
                                continue;
                            }
                            eprintln!(
                                "position {} of {:?}... in blocks of {size}: plain {want}, block pass {got}, the block pass prefers it by {margin:.3} (of a top logit {:.1})",
                                n * size + j,
                                &prompt[..20],
                                row[got as usize]
                            );
                            disagreements += 1;
                            let step = 2f32.powi(row[got as usize].abs().log2().floor() as i32 - 7);
                            worst = worst.max(margin / step);
                        }
                    }
                }
            }
        }
        eprintln!(
            "{disagreements} of {compared} positions differ (and {ties} exact ties), the largest margin {worst:.2} bfloat16 steps"
        );
        assert!(
            worst <= 10.0,
            "a disagreement that is not a near tie: the block pass prefers the other token by {worst} bfloat16 steps"
        );
        assert!(
            disagreements as f64 <= 0.03 * compared as f64,
            "the block pass disagrees with the plain loop at {disagreements} of {compared} positions, more than the 3% the shipped verify blocks come to"
        );
    }

    /// A sampled request is drafted too, the draws stay reproducible per seed, and the answer to a
    /// question with one right answer is right whatever the draws.
    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn a_sampled_request_is_drafted_and_reproducible() {
        let mut b = backend();
        b.set_draft_depth(Some(7));
        let ov = SamplingOverrides {
            seed: Some(21),
            ..Default::default()
        };
        let turns = [ChatTurn::User(
            "List the first 20 prime numbers, separated by commas.",
        )];
        let ask = |b: &mut Inkling2Backend| {
            b.clear_retained();
            let reply = b
                .chat(
                    &turns,
                    &[],
                    &[],
                    &ResolvedToolChoice::Auto,
                    300,
                    0.7,
                    1.0,
                    false,
                    &ov,
                    &mut |_| Ok(()),
                )
                .expect("chat");
            (reply.visible, b.last_run())
        };
        let (first, stats) = ask(&mut b);
        println!("{first:?}\n{stats:?}");
        assert!(
            first.contains("2, 3, 5, 7, 11, 13, 17, 19, 23, 29"),
            "{first}"
        );
        assert!(stats.spec.cycles > 0, "the request was not drafted");
        assert!(stats.spec.accepted > 0, "no draft was ever accepted");
        let (again, _) = ask(&mut b);
        assert_eq!(first, again, "the same seed, the same reply");
    }

    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
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
                        // The server's own wording: the engine's sinks do not say "stop sequence".
                        Err(anyhow!("__lumen_stop_sequence__"))
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

    /// Numeric regimes against the served one, over many prompts, greedy, each from a cold prefill
    /// (the retained caches are dropped between runs so one regime's caches never serve another's
    /// request). The reference is the default: composed mix, composed router, drafting on. Each
    /// variant turns one thing: the mix kernel, the router kernel, both, and drafting off. Drafting
    /// off is a weak control, not a measure of noise: on text the draft cannot predict the controller
    /// falls back to single-token steps, so it parts nowhere on these prompts. The test prints, per
    /// prompt, where each variant parts (`=` when it does not) and a tally; it asserts only that
    /// every run says something, because a part at a near tie is not a failure. How often a regime
    /// flips a choice is measured position by position in
    /// `a_kernel_flips_the_greedy_choice_only_at_near_ties_like_a_block_pass`.
    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn the_kernels_keep_the_greedy_text_over_many_prompts() {
        // (name, mix kernel, router kernel, drafting)
        const VARIANTS: [(&str, bool, bool, bool); 4] = [
            ("mix", true, false, true),
            ("route", false, true, true),
            ("both", true, true, true),
            ("control: no drafting", false, false, false),
        ];
        const MAX_NEW: usize = 160;
        let mut b = backend();
        let mut said = |mix: bool, route: bool, draft: bool, prompt: &str| -> String {
            b.clear_retained();
            let turns = [ChatTurn::User(prompt)];
            let reply = inkling2_draft::with(draft, || {
                super::super::mix_kernel::kernel::with(mix, || {
                    super::super::route_kernel::kernel::with(route, || {
                        greedy(&mut b, &turns, MAX_NEW)
                    })
                })
            });
            format!("{}{}", reply.reasoning, reply.visible)
        };
        let mut splits: Vec<Vec<usize>> = vec![Vec::new(); VARIANTS.len()];
        for (at, prompt) in PROMPTS.iter().enumerate() {
            let reference = said(false, false, true, prompt);
            assert!(!reference.is_empty(), "prompt {at} said nothing");
            let mut line = format!("[{at:>2}] of {:>4} chars:", reference.chars().count());
            for (v, (name, mix, route, draft)) in VARIANTS.iter().enumerate() {
                let text = said(*mix, *route, *draft, prompt);
                assert!(!text.is_empty(), "prompt {at} said nothing with {name}");
                if text == reference {
                    line.push_str(&format!("  {name}: ="));
                } else {
                    let split = reference
                        .chars()
                        .zip(text.chars())
                        .position(|(x, y)| x != y)
                        .unwrap_or_else(|| reference.chars().count().min(text.chars().count()));
                    line.push_str(&format!("  {name}: parts@{split}"));
                    splits[v].push(split);
                }
            }
            eprintln!("{line}");
        }
        for (v, (name, ..)) in VARIANTS.iter().enumerate() {
            let mut s = splits[v].clone();
            s.sort_unstable();
            eprintln!(
                "{name}: {} of {} parted from the reference; earliest chars {:?}, median {}",
                s.len(),
                PROMPTS.len(),
                &s[..s.len().min(5)],
                s.get(s.len() / 2).copied().unwrap_or(0)
            );
        }
    }

    /// How often each regime flips the greedy choice, position by position, and by how much. The
    /// plain loop's own continuation (kernels off, no drafting) of each prompt is fed back with each
    /// regime on: single steps with the mix kernel, the router kernel and both, and, as the baseline
    /// the project already ships, a pass over blocks of 2, 4 and 8 tokens as a verify pass has. At
    /// every position the regime's argmax is compared with the plain choice. Two top logits that are
    /// exactly equal (bfloat16 has few values, so it happens) are a tie, which the argmax breaks one
    /// way or the other, not a flip; a flip is measured in units of the bfloat16 step at the top
    /// logit (a few steps is a near tie, what
    /// `a_batched_pass_disagrees_with_single_steps_only_at_near_ties` accepts). Single steps with
    /// every kernel off must flip nothing: the determinism the other rows are measured against.
    /// It prints the table and every margin above 3 steps and asserts only that determinism.
    #[test]
    #[ignore = "needs the real checkpoint; set LUMEN_INKLING2_MODEL_DIR"]
    fn a_kernel_flips_the_greedy_choice_only_at_near_ties_like_a_block_pass() {
        const NEW_TOKENS: usize = 150;
        // (name, mix kernel, router kernel, block size)
        const REGIMES: [(&str, bool, bool, usize); 7] = [
            (
                "single steps, every kernel off (determinism)",
                false,
                false,
                1,
            ),
            ("single steps, mix kernel", true, false, 1),
            ("single steps, router kernel", false, true, 1),
            ("single steps, both kernels", true, true, 1),
            (
                "blocks of 2, kernels off (what drafting ships)",
                false,
                false,
                2,
            ),
            (
                "blocks of 4, kernels off (what drafting ships)",
                false,
                false,
                4,
            ),
            (
                "blocks of 8, kernels off (what drafting ships)",
                false,
                false,
                8,
            ),
        ];
        let mut b = backend();
        b.set_draft_depth(None);
        let ov = SamplingOverrides::default();
        let vocab = b.model.args().vocab_size;
        // Per regime: positions compared, exact ties, and the margin of every flip in bf16 steps.
        let mut compared = vec![0usize; REGIMES.len()];
        let mut ties = vec![0usize; REGIMES.len()];
        let mut flips: Vec<Vec<f32>> = vec![Vec::new(); REGIMES.len()];
        for prompt in PROMPTS {
            let ids = b
                .build_prompt(
                    &[ChatTurn::User(prompt)],
                    &[],
                    &ResolvedToolChoice::Auto,
                    false,
                    None,
                )
                .unwrap();
            let plain = super::super::mix_kernel::kernel::with(false, || {
                super::super::route_kernel::kernel::with(false, || {
                    b.generate(&ids, NEW_TOKENS, 0.0, 1.0, &ov).unwrap()
                })
            });
            let mut fed = vec![*ids.last().unwrap()];
            fed.extend_from_slice(&plain[..plain.len() - 1]);
            for (r, (_, mix, route, size)) in REGIMES.iter().enumerate() {
                super::super::mix_kernel::kernel::with(*mix, || {
                    super::super::route_kernel::kernel::with(*route, || {
                        let mut cache = b.model.make_cache();
                        b.model.advance(&ids[..ids.len() - 1], &mut cache).unwrap();
                        for (n, block) in fed.chunks(*size).enumerate() {
                            let (logits, _) = b.model.verify(block, &mut cache, &[5]).unwrap();
                            logits.eval().unwrap();
                            let rows = logits.as_slice::<f32>().to_vec();
                            for (j, row) in rows.chunks(vocab).enumerate() {
                                let want = plain[n * size + j];
                                let got = row
                                    .iter()
                                    .enumerate()
                                    .max_by(|a, c| a.1.total_cmp(c.1))
                                    .map(|(i, _)| i as u32)
                                    .unwrap();
                                compared[r] += 1;
                                if got == want {
                                    continue;
                                }
                                let margin = row[got as usize] - row[want as usize];
                                if margin <= 0.0 {
                                    ties[r] += 1;
                                } else {
                                    let step = 2f32
                                        .powi(row[got as usize].abs().log2().floor() as i32 - 7);
                                    flips[r].push(margin / step);
                                }
                            }
                        }
                    })
                });
            }
            eprintln!("done: {}...", &prompt[..prompt.len().min(40)]);
        }
        for (r, (name, ..)) in REGIMES.iter().enumerate() {
            let mut steps = flips[r].clone();
            steps.sort_by(f32::total_cmp);
            let beyond: Vec<String> = steps
                .iter()
                .filter(|s| **s > 3.0)
                .map(|s| format!("{s:.2}"))
                .collect();
            eprintln!(
                "{name}: {} flips ({:.2}%) and {} exact ties of {} positions; largest {:.2} bfloat16 steps; beyond 3 steps: [{}]",
                steps.len(),
                100.0 * steps.len() as f64 / compared[r].max(1) as f64,
                ties[r],
                compared[r],
                steps.last().copied().unwrap_or(0.0),
                beyond.join(", ")
            );
        }
        assert!(
            flips[0].is_empty(),
            "single steps with every kernel off must be deterministic, not flip {} positions",
            flips[0].len()
        );
    }
}
