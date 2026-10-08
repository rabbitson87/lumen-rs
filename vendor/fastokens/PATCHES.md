# Changes from upstream fastokens 0.3.2

This copy differs from the crates.io release in two places. Everything else,
including `LICENSE` and `NOTICES.txt`, is upstream's.

## `Cargo.toml`: no default features

Upstream's default feature is `hf-hub` (model downloads over HTTP). lumen-rs
loads `tokenizer.json` from disk, so the default is empty. The upstream examples
are not vendored, and their `[[example]]` entries are gone with them.

## Split cache: a reused match must not depend on bytes past the shared prefix

`src/pre_tokenizers/split.rs`, `pre_tokenize_pcre2_isolated` and the new
`reuse_limit`.

The thread-local split cache reuses the previous input's regex matches when the
new input shares a prefix of at least 4 KiB. Upstream reused every match with
`end < common_len`. That assumes a match is decided by bytes before its end, and
the patterns fastokens serves break the assumption:

- a whitespace run is decided by where the run stops: `\s+(?!\S)` backs off
  one space when a non-space follows, and `\s*[\r\n]+` backtracks to the run's
  last newline;
- an optional contraction suffix (`(?i:'s|'ll|…)?`) reads a few bytes past a
  word.

So the same string tokenized differently right after an input that diverged just
past a whitespace run. On every Qwen tokenizer: 5000 bytes of text plus 700
spaces, encoded right after the same text with an `x` after the spaces, split
the run as 59 + 2 where a fresh encode, and HF `tokenizers`, give one token.

Reuse now stops at the start of the whitespace run that reaches the first
differing byte, less a 16-byte margin. The cache only serves inputs of 4 KiB or
more, so the extra rescan is a few dozen bytes.

Regression tests, which fail with upstream's condition:

- `a_whitespace_match_ending_before_the_divergence_is_not_reused`
- `a_newline_run_is_decided_by_where_it_ends`
- `an_optional_suffix_reads_past_the_word`
