//! The one place a chat backend talks to its text tokenizer.
//!
//! Every chat-path encode and decode — Qwen's wrappers, Gemma's hand-ported
//! template, the opt-in jinja renderer — goes through [`TextTokenizer`]. Two
//! reasons to funnel them:
//!
//! * **Cost is visible.** Nothing used to time tokenization, and the prefill
//!   timers start after the prompt is built, so a request that tokenized the
//!   same 50K-token system+tools head three times looked exactly like one that
//!   did it once. [`TokenizeStats`] counts calls, tokens and wall time per
//!   tokenizer, and the engine prints the per-request delta.
//! * **The engine is swappable.** Call sites see ids and strings, never an HF
//!   `Encoding`, so a different encoder can sit behind this type without
//!   touching them (task 016 evaluates one).
//! * **Recurring pieces can be remembered.** An agent resends the same
//!   system+tools head and history every turn; under [`tokenize_memo`] those
//!   pieces come from a memo and a turn encodes only what is new.
//!
//! The embedding model and the FLUX.2 text encoder load their own HF
//! tokenizers on purpose: neither is on the chat latency path, and the
//! embedding model relies on `add_special_tokens = true` appending its EOS.
//!
//! Errors stay `tokenizers::Result`, so every caller keeps the exact message it
//! wraps them in today.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use tokenizers::processors::PostProcessorWrapper;

lumen_flags::flag! {
    /// Remember the encodes of long prompt pieces that recur across requests —
    /// the system+tools head and the history an agent resends every turn — so
    /// a turn encodes only what is new. Qwen prompts are cut right before each
    /// `<|im_start|>`, where HF splits them itself; other strings are
    /// remembered whole. Exact: the cut is used only for a tokenizer where it
    /// cannot change the ids, checked at load. Bounded at 64 MB. On by
    /// default since task 016's Gate 2: warm agentic turns at ~35K tokens
    /// reached their first token 38 ms sooner on Qwen3.5-9B (279 → 241 ms,
    /// Welch t 16.3) and 45 ms sooner on Gemma 4 (272 → 227 ms, t 16.2).
    pub tokenize_memo {
        env: "LUMEN_TOKENIZE_MEMO",
        default: true,
        kind: Optimization,
    }
}

/// Pieces shorter than this are cheap to encode and rarely recur verbatim.
const MIN_PIECE_BYTES: usize = 512;
/// Text plus ids held by the memo before it is cleared and refilled.
const MAX_MEMO_BYTES: usize = 64 << 20;
/// Where chat prompts are cut, when that is exact (see [`split_marker`]).
const TURN_MARKER: &str = "<|im_start|>";

/// Cumulative work done by one tokenizer, shared by every handle to it.
///
/// Relaxed atomics: each counter is read and written independently, and a
/// snapshot taken while another thread is mid-call may see that call's count
/// without its time. The engine only reads deltas around a request it serves
/// synchronously, where that cannot happen.
#[derive(Debug, Default)]
pub struct TokenizeStats {
    encode_calls: AtomicU64,
    encode_tokens: AtomicU64,
    encode_ns: AtomicU64,
    decode_calls: AtomicU64,
    decode_ns: AtomicU64,
}

/// A reading of [`TokenizeStats`]. Subtract two with [`Self::since`] to get
/// the work done in between.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenizeSnapshot {
    pub encode_calls: u64,
    pub encode_tokens: u64,
    pub encode_ns: u64,
    pub decode_calls: u64,
    pub decode_ns: u64,
}

impl TokenizeStats {
    pub fn snapshot(&self) -> TokenizeSnapshot {
        TokenizeSnapshot {
            encode_calls: self.encode_calls.load(Ordering::Relaxed),
            encode_tokens: self.encode_tokens.load(Ordering::Relaxed),
            encode_ns: self.encode_ns.load(Ordering::Relaxed),
            decode_calls: self.decode_calls.load(Ordering::Relaxed),
            decode_ns: self.decode_ns.load(Ordering::Relaxed),
        }
    }

    fn record_encode(&self, tokens: usize, started: Instant) {
        self.encode_calls.fetch_add(1, Ordering::Relaxed);
        self.encode_tokens
            .fetch_add(tokens as u64, Ordering::Relaxed);
        self.encode_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
    }

    fn record_decode(&self, started: Instant) {
        self.decode_calls.fetch_add(1, Ordering::Relaxed);
        self.decode_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
    }
}

impl TokenizeSnapshot {
    /// The work done between `earlier` and `self`.
    pub fn since(&self, earlier: &TokenizeSnapshot) -> TokenizeSnapshot {
        TokenizeSnapshot {
            encode_calls: self.encode_calls.saturating_sub(earlier.encode_calls),
            encode_tokens: self.encode_tokens.saturating_sub(earlier.encode_tokens),
            encode_ns: self.encode_ns.saturating_sub(earlier.encode_ns),
            decode_calls: self.decode_calls.saturating_sub(earlier.decode_calls),
            decode_ns: self.decode_ns.saturating_sub(earlier.decode_ns),
        }
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// A chat model's tokenizer: HF `tokenizers` underneath, ids and strings out.
pub struct TextTokenizer {
    hf: tokenizers::Tokenizer,
    stats: Arc<TokenizeStats>,
    memo: Mutex<Memo>,
    memo_cap: usize,
    /// The post-processor adds no ids, so `add_special_tokens` cannot change
    /// what a string encodes to and the memo may serve either.
    plain_post: bool,
    /// Encoding the pieces before each occurrence of this marker separately
    /// gives the whole string's ids. `None`: strings are remembered whole.
    split_before: Option<&'static str>,
}

/// Remembered encodes, keyed by the exact text.
#[derive(Default)]
struct Memo {
    entries: HashMap<Box<str>, Arc<[u32]>>,
    bytes: usize,
}

impl TextTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> tokenizers::Result<Self> {
        Ok(Self::from_hf(tokenizers::Tokenizer::from_file(path)?))
    }

    /// Wrap an already-built HF tokenizer (tests build theirs from a string).
    pub fn from_hf(hf: tokenizers::Tokenizer) -> Self {
        let plain_post = matches!(
            hf.get_post_processor(),
            None | Some(PostProcessorWrapper::ByteLevel(_))
        );
        let split_before = if plain_post { split_marker(&hf) } else { None };
        Self {
            hf,
            stats: Arc::default(),
            memo: Mutex::default(),
            memo_cap: MAX_MEMO_BYTES,
            plain_post,
            split_before,
        }
    }

    /// Token ids for `text`. `add_special_tokens` means what it means to HF:
    /// run the post-processor (BOS/EOS templates). Added tokens that appear
    /// literally in `text` are matched either way.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> tokenizers::Result<Vec<u32>> {
        let started = Instant::now();
        let memoizable = tokenize_memo::get()
            && text.len() >= MIN_PIECE_BYTES
            && (self.plain_post || !add_special_tokens);
        let ids = if memoizable {
            self.encode_memoized(text)?
        } else {
            self.hf.encode(text, add_special_tokens)?.get_ids().to_vec()
        };
        self.stats.record_encode(ids.len(), started);
        Ok(ids)
    }

    /// `text`'s ids from remembered pieces where possible. Called only when
    /// `add_special_tokens` cannot change them, so pieces encode without it.
    fn encode_memoized(&self, text: &str) -> tokenizers::Result<Vec<u32>> {
        let Some(marker) = self.split_before else {
            return Ok(self.piece(text)?.to_vec());
        };
        let mut ids = Vec::new();
        for piece in split_before(text, marker) {
            ids.extend_from_slice(&self.piece(piece)?);
        }
        Ok(ids)
    }

    fn piece(&self, piece: &str) -> tokenizers::Result<Arc<[u32]>> {
        if piece.len() < MIN_PIECE_BYTES {
            return Ok(self.hf.encode(piece, false)?.get_ids().into());
        }
        if let Some(ids) = self.memo().entries.get(piece) {
            return Ok(Arc::clone(ids));
        }
        let ids: Arc<[u32]> = self.hf.encode(piece, false)?.get_ids().into();
        let cost = piece.len() + ids.len() * size_of::<u32>();
        let mut memo = self.memo();
        if memo.bytes + cost > self.memo_cap {
            *memo = Memo::default();
        }
        if memo
            .entries
            .insert(piece.into(), Arc::clone(&ids))
            .is_none()
        {
            memo.bytes += cost;
        }
        Ok(ids)
    }

    fn memo(&self) -> MutexGuard<'_, Memo> {
        // A panic mid-insert leaves a whole entry or none; the map stays valid.
        self.memo.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> tokenizers::Result<String> {
        let started = Instant::now();
        let text = self.hf.decode(ids, skip_special_tokens)?;
        self.stats.record_decode(started);
        Ok(text)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.hf.id_to_token(id)
    }

    pub fn stats(&self) -> &Arc<TokenizeStats> {
        &self.stats
    }
}

/// The chat-turn marker prompts can be cut at without changing their ids.
///
/// HF pulls added tokens out of a string before it normalizes or pre-tokenizes
/// the rest, so the text on either side of one is already encoded on its own —
/// provided the token is matched without looking past its own bytes: no
/// `lstrip` (it would swallow whitespace from the piece before), no
/// `single_word` (it would read the character before), not `normalized` (it
/// would be matched after normalization), and no other added token that could
/// contain it or overlap it from the left (HF matches leftmost-first, so such a
/// token would claim those bytes instead).
fn split_marker(hf: &tokenizers::Tokenizer) -> Option<&'static str> {
    let added = hf.get_added_tokens_decoder();
    let marker_ok = added
        .values()
        .any(|t| t.content == TURN_MARKER && !t.lstrip && !t.single_word && !t.normalized);
    let clash = added.values().any(|t| {
        t.content != TURN_MARKER
            && (t.content.contains(TURN_MARKER)
                || (1..TURN_MARKER.len()).any(|k| t.content.ends_with(&TURN_MARKER[..k])))
    });
    (marker_ok && !clash).then_some(TURN_MARKER)
}

/// `text` cut right before every occurrence of `marker`. The pieces
/// concatenate back to `text`, and none is empty.
fn split_before<'a>(text: &'a str, marker: &str) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for (at, _) in text.match_indices(marker) {
        if at > start {
            pieces.push(&text[start..at]);
        }
        start = at;
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// WordLevel with a post-processor that appends `<eos>` — the shape of
    /// Qwen3-Embedding's, small enough to read.
    const EOS_TEMPLATE_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 3, "content": "<eos>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}}
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}}
            ],
            "special_tokens": {
                "<eos>": {"id": "<eos>", "ids": [3], "tokens": ["<eos>"]}
            }
        },
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<eos>": 3}
        }
    }"#;

    fn hf() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_str(EOS_TEMPLATE_TOKENIZER).expect("test tokenizer")
    }

    #[test]
    fn encode_and_decode_are_exactly_what_hf_returns() {
        let tok = TextTokenizer::from_hf(hf());
        let reference = hf();
        for add_special in [false, true] {
            for text in [
                "hello world",
                "world hello <eos> hello",
                "",
                "unknown words",
            ] {
                let want = reference
                    .encode(text, add_special)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let got = tok.encode(text, add_special).unwrap();
                assert_eq!(got, want, "{text:?} add_special={add_special}");
                for skip in [false, true] {
                    assert_eq!(
                        tok.decode(&got, skip).unwrap(),
                        reference.decode(&want, skip).unwrap(),
                        "decode {text:?} skip={skip}"
                    );
                }
            }
        }
    }

    /// The post-processor runs only when asked — the property a caller that
    /// relies on an appended EOS (the embedding model) cannot lose silently.
    #[test]
    fn add_special_tokens_controls_the_post_processor() {
        let tok = TextTokenizer::from_hf(hf());
        assert_eq!(tok.encode("hello world", false).unwrap(), vec![1, 2]);
        assert_eq!(tok.encode("hello world", true).unwrap(), vec![1, 2, 3]);
        assert_eq!(tok.id_to_token(3).as_deref(), Some("<eos>"));
    }

    #[test]
    fn stats_count_every_call_and_snapshots_subtract() {
        let tok = TextTokenizer::from_hf(hf());
        let before = tok.stats().snapshot();
        tok.encode("hello world", false).unwrap();
        tok.encode("hello", true).unwrap();
        tok.decode(&[1, 2], true).unwrap();
        let d = tok.stats().snapshot().since(&before);
        assert_eq!(d.encode_calls, 2);
        assert_eq!(d.encode_tokens, 2 + 2);
        assert_eq!(d.decode_calls, 1);
        // Subtracting in the wrong order saturates to zero instead of wrapping
        // into a delta of 2^64 calls.
        assert_eq!(
            before.since(&tok.stats().snapshot()),
            TokenizeSnapshot::default()
        );
    }

    /// The parts of Qwen's tokenizer the memo depends on: a turn marker added
    /// with every matching flag off (unless `lstrip`), an NFC normalizer and a
    /// ByteLevel post-processor that adds no ids. `extra` adds one more added
    /// token, to probe the guard.
    fn qwen_shaped(lstrip: bool, extra: &str) -> tokenizers::Tokenizer {
        let extra_added = if extra.is_empty() {
            String::new()
        } else {
            format!(
                r#",{{"id": 9, "content": {extra:?}, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true}}"#
            )
        };
        let extra_vocab = if extra.is_empty() {
            String::new()
        } else {
            format!(r#", {extra:?}: 9"#)
        };
        let json = format!(
            r#"{{
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {{"id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": {lstrip},
                  "rstrip": false, "normalized": false, "special": true}},
                {{"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false,
                  "rstrip": false, "normalized": false, "special": true}}{extra_added}
            ],
            "normalizer": {{"type": "NFC"}},
            "pre_tokenizer": {{"type": "Whitespace"}},
            "post_processor": {{"type": "ByteLevel", "add_prefix_space": false,
                                "trim_offsets": false, "use_regex": true}},
            "decoder": null,
            "model": {{"type": "WordLevel", "unk_token": "<unk>", "vocab": {{
                "<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "hello": 3, "world": 4,
                "system": 5, "user": 6, "assistant": 7, "café": 8{extra_vocab}
            }}}}
        }}"#
        );
        tokenizers::Tokenizer::from_str(&json).expect("qwen-shaped tokenizer")
    }

    /// A conversation grown turn by turn, as an agent's is, so every call
    /// after the first finds most of its pieces remembered. Returns how many
    /// turns were checked.
    fn assert_memo_exact(tok: &TextTokenizer, reference: &tokenizers::Tokenizer) -> usize {
        let words = [
            "hello",
            "world",
            "system",
            "user",
            "assistant",
            "café",
            "cafe\u{301}",
            "<|im_end|>",
            "\n",
            " ",
            "  ",
            "\t",
            "unknown",
        ];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut convo = String::new();
        let mut turns = 0;
        tokenize_memo::with(true, || {
            for turn in 0..40 {
                convo.push_str("<|im_start|>");
                // Now and then two markers back to back: an empty turn.
                if next() % 8 == 0 {
                    convo.push_str("<|im_start|>");
                }
                let goal = convo.len() + MIN_PIECE_BYTES + (next() % 600) as usize;
                while convo.len() < goal {
                    convo.push_str(words[(next() % words.len() as u64) as usize]);
                }
                for add_special in [false, true] {
                    let want = reference.encode(convo.as_str(), add_special).unwrap();
                    let got = tok.encode(&convo, add_special).unwrap();
                    assert_eq!(got, want.get_ids(), "turn {turn} add_special={add_special}");
                }
                turns += 1;
            }
        });
        turns
    }

    #[test]
    fn the_memo_returns_exactly_what_hf_does() {
        let tok = TextTokenizer::from_hf(qwen_shaped(false, ""));
        assert_eq!(
            tok.split_before,
            Some(TURN_MARKER),
            "the Qwen shape is cut at turns"
        );
        assert_eq!(assert_memo_exact(&tok, &qwen_shaped(false, "")), 40);
        assert!(
            tok.memo().bytes > 0,
            "nothing was remembered: the memo never ran"
        );
    }

    #[test]
    fn a_marker_that_reads_its_neighbours_is_never_cut_at() {
        for (why, hf) in [
            (
                "lstrip swallows the whitespace before it",
                qwen_shaped(true, ""),
            ),
            (
                "a token ending in its prefix overlaps it",
                qwen_shaped(false, "x<|"),
            ),
            (
                "a token containing it",
                qwen_shaped(false, "<|im_start|>user"),
            ),
        ] {
            let tok = TextTokenizer::from_hf(hf);
            assert_eq!(tok.split_before, None, "{why}");
        }
        // Remembered whole instead, which is exact for any tokenizer.
        let tok = TextTokenizer::from_hf(qwen_shaped(true, ""));
        assert_eq!(assert_memo_exact(&tok, &qwen_shaped(true, "")), 40);
    }

    /// A post-processor that adds ids (the embedding model's EOS) runs exactly
    /// as before: the memo only serves calls whose ids it cannot change.
    #[test]
    fn a_post_processor_that_adds_ids_still_runs() {
        let tok = TextTokenizer::from_hf(hf());
        assert_eq!(tok.split_before, None);
        let text = "hello world ".repeat(60);
        tokenize_memo::with(true, || {
            for add_special in [false, true, true] {
                let want = hf().encode(text.as_str(), add_special).unwrap();
                assert_eq!(tok.encode(&text, add_special).unwrap(), want.get_ids());
            }
        });
        assert_eq!(
            tok.encode(&text, true).unwrap().last(),
            Some(&3),
            "EOS kept"
        );
    }

    #[test]
    fn the_memo_stays_under_its_cap() {
        let mut tok = TextTokenizer::from_hf(qwen_shaped(false, ""));
        tok.memo_cap = 4096;
        tokenize_memo::with(true, || {
            for i in 0..50 {
                let text = format!("<|im_start|>{}", format!("hello{i} world ").repeat(60));
                tok.encode(&text, false).unwrap();
                assert!(tok.memo().bytes <= 4096, "{} bytes", tok.memo().bytes);
            }
        });
    }

    /// Exact on the tokenizer the cut exists for, over the prompt shapes lumen
    /// renders: a long system+tools head, tool calls and results, Korean,
    /// code, JSON and whitespace runs.
    #[test]
    #[ignore = "requires a real Qwen tokenizer.json; set LUMEN_QWEN35_MODEL_DIR"]
    fn the_memo_is_exact_on_a_real_qwen_tokenizer() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_MODEL_DIR") else {
            eprintln!("skip: set LUMEN_QWEN35_MODEL_DIR to a Qwen checkpoint");
            return;
        };
        let path = Path::new(&dir).join("tokenizer.json");
        let reference = tokenizers::Tokenizer::from_file(&path).expect("tokenizer.json");
        let tok = TextTokenizer::from_file(&path).expect("tokenizer.json");
        assert_eq!(
            tok.split_before,
            Some(TURN_MARKER),
            "a Qwen tokenizer is cut at turns"
        );
        let docs = include_str!("../../../docs/maintainer-workflow.md");
        let file = include_str!("prefix_cache.rs");
        let mut convo = format!(
            "<|im_start|>system\n{docs}\n\n# Tools\n\n<tools>\n{{\"type\": \"function\", \
             \"function\": {{\"name\": \"read_file\", \"parameters\": {{\"path\": \
             {{\"type\": \"string\"}}}}}}}}\n</tools><|im_end|>\n"
        );
        let turns = [
            "<|im_start|>user\nRead crates/lumen-mlx/src/prefix_cache.rs for me.<|im_end|>\n".to_string(),
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": \
             {\"path\": \"prefix_cache.rs\"}}\n</tool_call><|im_end|>\n"
                .to_string(),
            format!("<|im_start|>user\n<tool_response>\n{file}\n</tool_response><|im_end|>\n"),
            "<|im_start|>user\n이 파일에서 캐시 적중은 어떻게 결정되나요?   \t\n<|im_end|>\n".to_string(),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n`longest_hit` 가 결정합니다.<|im_end|>\n"
                .to_string(),
            "<|im_start|>user\nAnd the boundary?<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ];
        tokenize_memo::with(true, || {
            for (n, turn) in turns.iter().enumerate() {
                convo.push_str(turn);
                for add_special in [false, true] {
                    let want = reference.encode(convo.as_str(), add_special).unwrap();
                    let got = tok.encode(&convo, add_special).unwrap();
                    assert_eq!(got, want.get_ids(), "turn {n} add_special={add_special}");
                }
            }
        });
    }
}
