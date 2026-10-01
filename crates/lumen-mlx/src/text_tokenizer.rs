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
//!
//! The embedding model and the FLUX.2 text encoder load their own HF
//! tokenizers on purpose: neither is on the chat latency path, and the
//! embedding model relies on `add_special_tokens = true` appending its EOS.
//!
//! Errors stay `tokenizers::Result`, so every caller keeps the exact message it
//! wraps them in today.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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
}

impl TextTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> tokenizers::Result<Self> {
        Ok(Self::from_hf(tokenizers::Tokenizer::from_file(path)?))
    }

    /// Wrap an already-built HF tokenizer (tests build theirs from a string).
    pub fn from_hf(hf: tokenizers::Tokenizer) -> Self {
        Self {
            hf,
            stats: Arc::default(),
        }
    }

    /// Token ids for `text`. `add_special_tokens` means what it means to HF:
    /// run the post-processor (BOS/EOS templates). Added tokens that appear
    /// literally in `text` are matched either way.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> tokenizers::Result<Vec<u32>> {
        let started = Instant::now();
        let ids = self.hf.encode(text, add_special_tokens)?.get_ids().to_vec();
        self.stats.record_encode(ids.len(), started);
        Ok(ids)
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
}
