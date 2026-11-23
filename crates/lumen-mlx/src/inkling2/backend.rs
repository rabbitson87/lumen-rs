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
use lumen_core::sampling::{SamplingConfig, Xorshift64, sample_from_logits};
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
    /// (`mtp.safetensors`) proposes a block of tokens and the model checks the block in one pass,
    /// for greedy requests (temperature 0). Off by default: it is on only where it measured faster
    /// than the plain loop, and a pass over a block does not round exactly like single-token steps
    /// in bfloat16, so a continuation can differ from the plain loop's where two tokens are near
    /// ties (hence a behavior flag). The draft head's 0.6 GB load with the first request.
    pub(crate) inkling2_draft {
        env: "LUMEN_INKLING2_DRAFT",
        default: false,
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
        Ok(self
            .prompt_with_images(turns, &[], tools, tool_choice, thinking, named)?
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
        let prompt = self.prompt_with_images(
            turns,
            images,
            tools,
            tool_choice,
            thinking,
            named_effort(ov),
        )?;
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
        let mut stopped_by_sink = false;
        let generated = self.generate_with(
            &prompt,
            &sources,
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
        self.generate_with(
            &Prompt::plain(prompt),
            &[],
            max_new_tokens,
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
            prompt.ids.len() + max_new_tokens.min(1) <= self.max_context,
            "a prompt of {} tokens is past this backend's context of {}",
            prompt.ids.len(),
            self.max_context
        );
        let speculate =
            sampling.is_none() && self.draft_depth.unwrap_or(0) > 0 && self.ensure_draft()?;
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
                &mut |anchor, rows, want, _| {
                    if want == 0 {
                        if let Some(rows) = rows {
                            draft.extend(&mut context, rows)?;
                        }
                        return Ok(Vec::new());
                    }
                    Ok(draft.draft(&mut context, rows, anchor, &self.model)?.tokens)
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
    propose: &mut dyn FnMut(u32, Option<&Array>, usize, usize) -> Result<Vec<u32>>,
) -> Result<()> {
    let mut pending: Option<Array> = None;

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

    let mut anchor = pick(first_logits, seen, None, &mut Xorshift64::new(0))?;
    let mut flow = emit(anchor, generated)?;
    while matches!(flow, Flow::More) {
        let remaining = plan.max_new_tokens - generated.len();
        let k = plan.depth.min(remaining.saturating_sub(1));

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
        let choice = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false)?;
        mlx_rs::transforms::eval([&choice, &taps])?;
        let chosen: Vec<u32> = choice
            .as_dtype(mlx_rs::Dtype::Uint32)?
            .as_slice::<u32>()
            .to_vec();
        stats.verify_ms += verified_at.elapsed().as_secs_f64() * 1e3;

        let agreed = (0..k).take_while(|&i| block[i] == chosen[i]).count();
        stats.cycles += 1;
        stats.drafted += k;
        stats.accepted += agreed;

        // Emit the model's own tokens in order, until one ends the turn.
        let mut kept = 0;
        for &token in &chosen[..=agreed] {
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

    /// The speculative loop with a drafter that proposes `oracle[at..at + want]`, with the entry at
    /// block position `wrong` (if any) replaced by a token that cannot be right.
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
        assert!(stats.reused_tokens > 100, "reused {}", stats.reused_tokens);
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
