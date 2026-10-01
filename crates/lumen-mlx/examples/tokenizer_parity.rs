//! Does `fastokens` produce exactly the ids HF `tokenizers` does on our
//! tokenizers, and how much faster is it on this machine?
//!
//! Task 016, Phase 0c. fastokens (crates.io, Crusoe) is a BPE-only encoder that
//! reads the same `tokenizer.json` and claims 10x+ over HF. lumen keys session
//! reuse, the prefix cache, disk KV and grammar replay on token ids, so one
//! differing id is a silent cache miss or a dropped grammar: the bar is
//! bit-identical, not close.
//!
//! For each `tokenizer.json` (default: the four distinct pipelines on this
//! machine) it compares `HF.encode(text, add_special)` with
//! `fastokens.encode_with_special_tokens(text, add_special)` — not
//! `fastokens::Tokenizer::encode`, which hard-codes `add_special = false` —
//! over:
//!
//! * agentic prompts: lumen's own Qwen system+tools head
//!   (`render_tools_system_block`) plus history with tool calls, file-sized
//!   tool results and thinking blocks, or Gemma 4's turn framing;
//! * multilingual text, combining marks, emoji sequences, NFC-sensitive text;
//! * whitespace/punctuation runs, and single regex matches longer than
//!   fastokens' 1 KB chunk overlap placed exactly across its parallel-chunk
//!   boundaries — inside plain text and inside a long tool-result segment,
//!   since the two take different paths. Its past id bugs (#59, #67, #69)
//!   lived in the repair step at those boundaries;
//! * a shared-prefix sequence that exercises its thread-local regex-match
//!   cache (inputs that share ≥ 4 KB with the previous one reuse its matches);
//! * special-token strings inside content, whole and partial;
//! * every Unicode scalar value, in three contexts;
//! * seeded random text over all of the above alphabets.
//!
//! fastokens is loaded with `from_json`: its `from_file` also merges a sibling
//! `tokenizer_config.json`, which HF does not. `.cargo/config.toml` sets
//! `PCRE2_SYS_STATIC=1` so PCRE2 is the vendored build, not Homebrew's.
//!
//! ```text
//! cargo run --release -p lumen-mlx --example tokenizer_parity
//! cargo run --release -p lumen-mlx --example tokenizer_parity -- a/tokenizer.json b/tokenizer.json
//! # timing only, fastokens on one thread (algorithm vs parallelism):
//! FASTOKENS_BPE_THREADS=1 RAYON_NUM_THREADS=1 \
//!   cargo run --release -p lumen-mlx --example tokenizer_parity -- --bench-only
//! ```
//!
//! Exits 1 if any id differs outside the code-point sweep. Sweep divergences
//! are printed with their code points and decided on separately (plan T5).
//!
//! ## Result on record (fastokens 0.3.2 vs tokenizers 0.23.1, M3 Max, 14 cores)
//!
//! **Gemma 4 is clean**: zero differing ids in every section, zero divergent
//! code points.
//!
//! **Every Qwen pipeline (3.5/3.8, 3.6, Qwen3-Embedding) has one real bug: ids
//! depend on what the same thread encoded before.** The shared-prefix
//! sequence fails, and the two-call repro pins it: 5000 B of text plus 700
//! spaces encodes correctly on its own, and differently right after an input
//! that had `x` where it continues its whitespace run (one more space, a
//! newline, a tab). fastokens' thread-local regex-match cache reuses the
//! previous input's matches that end before the first differing byte, but
//! `\s+(?!\S)` places a whitespace match's end by looking one byte past it,
//! so a divergence exactly there leaves a stale split (59 + 2 spaces where HF
//! has one token). The cache only serves inputs with no added tokens of 4 KiB
//! or more: lumen's chat prompts carry `<|im_start|>` and take the generic
//! path, but raw `/v1/completions` prompts do not.
//!
//! Everything else is identical on all four: agentic prompts, 8,000 seeded
//! random cases, multilingual text, whitespace runs, and every long-match
//! placement across the parallel-chunk boundaries on both paths. The
//! code-point sweep differs on four combining marks before U+0301 (U+1AEB,
//! U+1DF6, U+1E4EC, U+1E4ED): fastokens' NFC (ICU, newer Unicode) reorders
//! them by combining class and HF's tables predate them.
//!
//! Speed, median ms per encode of an agentic prompt:
//!
//! | tokenizer | tokens | HF | fastokens (14 threads) | fastokens (1 thread) |
//! |---|---|---|---|---|
//! | Qwen3.8 | 30,799 | 18.4 | 1.34 (13.7x) | 1.54 (10.9x) |
//! | Qwen3.8 | 107,963 | 80.4 | 5.19 (15.5x) | 6.37 (10.5x) |
//! | Gemma 4 | 31,471 | 12.8 | 0.90 (14.1x) | 1.07 (11.9x) |
//! | Gemma 4 | 116,347 | 61.7 | 3.78 (16.3x) | 4.92 (12.3x) |
//!
//! Gemma's piecewise render (93 encodes, as lumen's renderer does it): 11.3 ms
//! vs 1.16 ms. Loading is the one place fastokens is slower: 0.67-1.2 s vs
//! 0.15-0.3 s for HF.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use lumen_mlx::chat_io::ToolDef;
use serde_json::{Value, json};

fn main() -> ExitCode {
    let paths = tokenizer_paths();
    if paths.is_empty() {
        eprintln!("no tokenizer.json found; pass paths as arguments");
        return ExitCode::from(2);
    }
    let repo = RepoText::load();
    println!(
        "fastokens threads: FASTOKENS_BPE_THREADS={} RAYON_NUM_THREADS={} cores={}",
        std::env::var("FASTOKENS_BPE_THREADS").unwrap_or_else(|_| "default".into()),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    );

    // `--bench-only` skips the parity sections, for re-timing under other
    // thread settings without repeating several minutes of comparisons.
    let bench_only = std::env::args().any(|a| a == "--bench-only");
    let mut realistic_mismatches = 0;
    for path in &paths {
        let Some(e) = Engines::load(path) else {
            continue;
        };
        if bench_only {
            e.bench(&repo);
        } else {
            realistic_mismatches += e.run(&repo);
        }
    }
    if bench_only {
        println!("\n(--bench-only: parity was not checked)");
        ExitCode::SUCCESS
    } else if realistic_mismatches > 0 {
        println!("\nFAIL: {realistic_mismatches} mismatching case(s) outside the code-point sweep");
        ExitCode::FAILURE
    } else {
        println!("\nPASS: every non-sweep case produced identical ids");
        ExitCode::SUCCESS
    }
}

/// The four distinct pipelines on this machine (Qwen3.8/3.5 regex, Qwen3.6
/// regex with `\p{M}`, Qwen3-Embedding's EOS post-processor, Gemma 4), or the
/// paths given on the command line.
fn tokenizer_paths() -> Vec<PathBuf> {
    let args: Vec<PathBuf> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();
    if !args.is_empty() {
        return args;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    [
        "models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed/tokenizer.json",
        "models/Qwen3.6-27B-MTPLX-Speed/tokenizer.json",
        "models/qwen3-embedding-0.6b-8bit/tokenizer.json",
        "models/mlx-community--gemma-4-26b-a4b-it-4bit/tokenizer.json",
    ]
    .iter()
    .map(|rel| Path::new(&home).join(rel))
    .filter(|p| p.is_file())
    .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Qwen,
    Gemma,
}

struct Engines {
    label: String,
    family: Family,
    hf: tokenizers::Tokenizer,
    fast: fastokens::Tokenizer,
}

/// Outcome of comparing one input.
enum Verdict {
    Same,
    Differs {
        hf: Vec<u32>,
        fast: Vec<u32>,
    },
    /// fastokens returned an error where HF produced ids (plan T6).
    FastError(String),
}

impl Engines {
    fn load(path: &Path) -> Option<Self> {
        let label = path.parent().and_then(|p| p.file_name()).map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let raw = std::fs::read_to_string(path).ok()?;

        let t0 = Instant::now();
        let hf = match tokenizers::Tokenizer::from_file(path) {
            Ok(t) => t,
            Err(e) => {
                println!("\n=== {label}: HF failed to load ({e}) — skipped");
                return None;
            }
        };
        let hf_ms = ms(t0);
        let t0 = Instant::now();
        let json: Value = serde_json::from_str(&raw).ok()?;
        let fast = match fastokens::Tokenizer::from_json(json) {
            Ok(t) => t,
            Err(e) => {
                println!("\n=== {label}: fastokens refused to load: {e}");
                return None;
            }
        };
        let fast_ms = ms(t0);

        let family = if hf.token_to_id("<|turn>").is_some() {
            Family::Gemma
        } else {
            Family::Qwen
        };
        println!(
            "\n=== {label} ({}) — load: HF {hf_ms:.0} ms, fastokens {fast_ms:.0} ms",
            match family {
                Family::Qwen => "Qwen framing",
                Family::Gemma => "Gemma framing",
            }
        );
        Some(Self {
            label,
            family,
            hf,
            fast,
        })
    }

    fn compare(&self, text: &str, add_special: bool) -> Verdict {
        let hf = self
            .hf
            .encode(text, add_special)
            .expect("HF encode")
            .get_ids()
            .to_vec();
        match self.fast.encode_with_special_tokens(text, add_special) {
            Ok(fast) if fast == hf => Verdict::Same,
            Ok(fast) => Verdict::Differs { hf, fast },
            Err(e) => Verdict::FastError(e.to_string()),
        }
    }

    /// Returns the number of mismatching cases outside the code-point sweep.
    fn run(&self, repo: &RepoText) -> usize {
        let mut mismatches = 0;
        for (section, texts) in corpus(self.family, repo) {
            mismatches += self.check_section(section, &texts);
        }
        mismatches += self.check_prefix_reuse(repo);
        mismatches += self.check_random(4000);
        self.sweep_code_points();
        self.check_decode(repo);
        self.bench(repo);
        mismatches
    }

    fn check_section(&self, section: &str, texts: &[String]) -> usize {
        let mut bad = 0;
        let mut first: Option<(String, bool)> = None;
        let bytes: usize = texts.iter().map(String::len).sum();
        for text in texts {
            for add_special in [false, true] {
                match self.compare(text, add_special) {
                    Verdict::Same => {}
                    Verdict::Differs { .. } | Verdict::FastError(_) => {
                        bad += 1;
                        if first.is_none() {
                            first = Some((text.clone(), add_special));
                        }
                    }
                }
            }
        }
        println!(
            "  {section:<28} {:>5} cases {:>9} bytes  mismatches {bad}",
            texts.len() * 2,
            bytes
        );
        if let Some((text, add_special)) = first {
            self.explain(&text, add_special);
        }
        bad
    }

    /// Feed fastokens a sequence of inputs that share long prefixes with the
    /// previous one, so its thread-local regex-match cache is reused, then
    /// diverge at awkward places: inside a long letter run, inside a
    /// whitespace run, one byte after a match end.
    fn check_prefix_reuse(&self, repo: &RepoText) -> usize {
        let base = format!(
            "{}{}{}{}",
            repo.ascii(6_000),
            " ".repeat(700),
            "x".repeat(1_500),
            repo.ascii(14_000)
        );
        let cuts = [
            4_096, 6_000, 6_350, 6_699, 6_700, 6_701, 7_000, 7_900, 8_200, 15_000,
        ];
        let mut texts = vec![("base".to_string(), base.clone())];
        for (i, &cut) in cuts.iter().enumerate() {
            let cut = floor_char(&base, cut);
            let suffix = ["Z", " tail", "\n\nmore", "  "][i % 4];
            texts.push((
                format!("base[..{cut}] + {suffix:?}"),
                format!("{}{suffix}", &base[..cut]),
            ));
            texts.push(("base".to_string(), base.clone()));
            texts.push((
                format!("base + \" extended {i}\""),
                format!("{base} extended {i}"),
            ));
        }
        // Stateful: the outcome depends on the previous call, so a mismatch
        // is reported as the (previous, current) pair rather than shrunk —
        // the shrinker's own calls would replace the state it depends on.
        let mut bad = 0;
        let mut first: Option<String> = None;
        for add_special in [false, true] {
            for (k, (what, text)) in texts.iter().enumerate() {
                if !matches!(self.compare(text, add_special), Verdict::Same) {
                    bad += 1;
                    let prev = if k == 0 {
                        "(start)"
                    } else {
                        texts[k - 1].0.as_str()
                    };
                    first.get_or_insert_with(|| format!("{prev}  ->  {what}"));
                }
            }
        }
        println!(
            "  {:<28} {:>5} cases {:>9} bytes  mismatches {bad}",
            "shared-prefix sequence",
            texts.len() * 2,
            texts.iter().map(|t| t.1.len()).sum::<usize>()
        );
        if let Some(pair) = first {
            println!("      first mismatch, previous -> current: {pair}");
        }
        bad + self.check_cache_lookahead(repo)
    }

    /// The shared-prefix mismatches reduced to two calls. fastokens reuses the
    /// previous input's regex matches that end before the first differing
    /// byte (`end < common_len`), but Qwen's `\s+(?!\S)` decides where a
    /// whitespace match ends by looking one byte *past* that end. If the two
    /// inputs diverge exactly there, the reused match is stale. Each case is
    /// encoded fresh first, then again right after `previous`.
    fn check_cache_lookahead(&self, repo: &RepoText) -> usize {
        let filler = repo.ascii(5_000);
        let spaces = " ".repeat(700);
        let previous = format!("{filler}{spaces}x{}", repo.ascii(2_000));
        let cases = [
            (
                "run grows by one space, then ends",
                format!("{filler}{spaces} "),
            ),
            (
                "run is followed by a newline",
                format!("{filler}{spaces}\n"),
            ),
            ("run is followed by a tab", format!("{filler}{spaces}\tx")),
            (
                "control: run followed by a letter",
                format!("{filler}{spaces}y"),
            ),
        ];
        let mut bad = 0;
        println!(
            "      cache lookahead repro (5000 B filler + 700 spaces; previous has 'x' next):"
        );
        for (what, current) in &cases {
            // An input sharing no 4 KiB prefix replaces the cached state.
            self.fast
                .encode_with_special_tokens("reset", false)
                .expect("fast");
            let fresh = matches!(self.compare(current, false), Verdict::Same);
            self.fast
                .encode_with_special_tokens(&previous, false)
                .expect("fast");
            let after = self.compare(current, false);
            let after_same = matches!(after, Verdict::Same);
            if !fresh || !after_same {
                bad += 1;
            }
            println!(
                "        {what:<36} fresh: {}   after previous: {}",
                if fresh { "same" } else { "DIFFERS" },
                if after_same { "same" } else { "DIFFERS" }
            );
            if let Verdict::Differs { hf, fast } = after {
                let at = hf.iter().zip(&fast).take_while(|(a, b)| a == b).count();
                let show = |ids: &[u32]| -> Vec<String> {
                    ids[at..]
                        .iter()
                        .take(4)
                        .map(|&id| {
                            format!("{id}:{:?}", self.hf.id_to_token(id).unwrap_or_default())
                        })
                        .collect()
                };
                println!(
                    "          from token {at}: HF {:?} / fast {:?}",
                    show(&hf),
                    show(&fast)
                );
            }
        }
        bad
    }

    fn check_random(&self, cases: usize) -> usize {
        let mut rng = Rng(0x5EED_016F_A570_C3E5);
        let texts: Vec<String> = (0..cases)
            .map(|i| {
                let len = if i % 50 == 0 {
                    40_000
                } else {
                    1 + rng.below(3_000)
                };
                random_text(&mut rng, len)
            })
            .collect();
        self.check_section("seeded random", &texts)
    }

    /// Every scalar value in three contexts — between letters, in a run, and
    /// before a combining acute (NFC composition) — batched 64 at a time and
    /// drilled down to single code points only where a batch disagrees.
    fn sweep_code_points(&self) {
        let forms: [(&str, fn(char) -> String); 3] = [
            ("a{c}b", |c| format!("a{c}b ")),
            ("run", |c| c.to_string()),
            ("{c}+U+0301", |c| format!("{c}\u{301} ")),
        ];
        let all: Vec<char> = (0..=0x10FFFFu32).filter_map(char::from_u32).collect();
        let t0 = Instant::now();
        let mut divergent: Vec<(&str, char)> = Vec::new();
        for (name, form) in forms {
            for batch in all.chunks(64) {
                let text: String = batch.iter().map(|&c| form(c)).collect();
                if matches!(self.compare(&text, false), Verdict::Same) {
                    continue;
                }
                for &c in batch {
                    if !matches!(self.compare(&form(c), false), Verdict::Same) {
                        divergent.push((name, c));
                    }
                }
            }
        }
        println!(
            "  code-point sweep             {} scalars x 3 forms ({:.0} s)  divergent {}",
            all.len(),
            t0.elapsed().as_secs_f64(),
            divergent.len()
        );
        for (name, c) in divergent.iter().take(40) {
            println!("      {name:<12} U+{:04X} {c:?}", *c as u32);
        }
        if divergent.len() > 40 {
            println!("      ... {} more", divergent.len() - 40);
        }
    }

    /// Report-only: decode HF's ids with both engines. fastokens would only
    /// ever be used to encode (plan Decision 3), so this does not gate.
    fn check_decode(&self, repo: &RepoText) {
        let mut cases = 0;
        let mut differ = 0;
        for (_, texts) in corpus(self.family, repo) {
            for text in texts.iter().take(40) {
                let ids = self
                    .hf
                    .encode(text.as_str(), false)
                    .expect("HF")
                    .get_ids()
                    .to_vec();
                for skip in [false, true] {
                    cases += 1;
                    let h = self.hf.decode(&ids, skip).expect("HF decode");
                    if self.fast.decode(&ids, skip).ok().as_deref() != Some(h.as_str()) {
                        differ += 1;
                    }
                }
            }
        }
        println!("  decode (report only)         {cases:>5} cases  differ {differ}");
    }

    /// Median wall time of each engine on agentic prompts of several sizes.
    /// "Repeat" re-encodes the same text (what lumen's repeated head encodes
    /// look like); "fresh" changes the first bytes each run so no input-level
    /// reuse applies. Both engines keep per-word BPE caches either way.
    fn bench(&self, repo: &RepoText) {
        println!("  speed (median of 7 after 1 warm-up; ms)");
        println!(
            "    {:>8}  {:>9} {:>9} {:>7}  {:>9} {:>9} {:>7}",
            "tokens", "HF rep", "fast rep", "x", "HF fresh", "fast fr", "x"
        );
        // The system+tools head alone is ~27K tokens; history grows it from there.
        let head = agentic(self.family, repo, 0, 0).len();
        for history_bytes in [0, 40_000, 120_000, 300_000] {
            let text = agentic(self.family, repo, head + history_bytes, 0);
            let tokens = self.hf.encode(text.as_str(), false).expect("HF").len();
            let hf_rep = median_ms(|_| {
                self.hf.encode(text.as_str(), false).expect("HF");
            });
            let fast_rep = median_ms(|_| {
                self.fast
                    .encode_with_special_tokens(&text, false)
                    .expect("fast");
            });
            let fresh = |run: usize| format!("run {run} {text}");
            let hf_fresh = median_ms(|run| {
                self.hf.encode(fresh(run), false).expect("HF");
            });
            let fast_fresh = median_ms(|run| {
                self.fast
                    .encode_with_special_tokens(&fresh(run), false)
                    .expect("fast");
            });
            println!(
                "    {tokens:>8}  {hf_rep:>9.2} {fast_rep:>9.2} {:>6.1}x  {hf_fresh:>9.2} {fast_fresh:>9.2} {:>6.1}x",
                hf_rep / fast_rep,
                hf_fresh / fast_fresh
            );
        }
        if self.family == Family::Gemma {
            // lumen's Gemma renderer encodes the text between special tokens
            // one piece at a time (~100 calls per render) and splices the
            // special ids in itself.
            let text = agentic(self.family, repo, 80_000, 0);
            let pieces: Vec<&str> = split_at_specials(&text, GEMMA_SPECIALS);
            let hf_ms = median_ms(|_| {
                for p in &pieces {
                    self.hf.encode(*p, false).expect("HF");
                }
            });
            let fast_ms = median_ms(|_| {
                for p in &pieces {
                    self.fast
                        .encode_with_special_tokens(p, false)
                        .expect("fast");
                }
            });
            let same = pieces.iter().all(|p| {
                self.hf.encode(*p, false).expect("HF").get_ids()
                    == self
                        .fast
                        .encode_with_special_tokens(p, false)
                        .expect("fast")
            });
            println!(
                "    piecewise render: {} pieces  HF {hf_ms:.2}  fast {fast_ms:.2}  {:.1}x  ids identical: {same}",
                pieces.len(),
                hf_ms / fast_ms
            );
        }
    }

    /// Shrink a mismatching input to a small reproduction and print where the
    /// two id sequences part ways.
    fn explain(&self, text: &str, add_special: bool) {
        let differs = |t: &str| !matches!(self.compare(t, add_special), Verdict::Same);
        let mut cur: Vec<char> = text.chars().collect();
        let mut chunk = cur.len() / 2;
        let mut budget = 3_000;
        while chunk >= 1 && budget > 0 {
            let mut i = 0;
            let mut removed = false;
            while i < cur.len() && budget > 0 {
                let end = (i + chunk).min(cur.len());
                let candidate: String = cur[..i].iter().chain(&cur[end..]).collect();
                budget -= 1;
                if !candidate.is_empty() && differs(&candidate) {
                    cur.drain(i..end);
                    removed = true;
                } else {
                    i += chunk;
                }
            }
            if !removed {
                chunk /= 2;
            }
        }
        let small: String = cur.iter().collect();
        println!(
            "      first mismatch (minimized to {} chars, add_special={add_special}): {:?}",
            small.chars().count(),
            truncate(&small, 300)
        );
        match self.compare(&small, add_special) {
            Verdict::Differs { hf, fast } => {
                let at = hf.iter().zip(&fast).take_while(|(a, b)| a == b).count();
                let lo = at.saturating_sub(3);
                let show = |ids: &[u32]| -> Vec<String> {
                    ids.iter()
                        .skip(lo)
                        .take(8)
                        .map(|&id| {
                            format!("{id}:{:?}", self.hf.id_to_token(id).unwrap_or_default())
                        })
                        .collect()
                };
                println!("      HF   [{lo}..] {:?}", show(&hf));
                println!("      fast [{lo}..] {:?}", show(&fast));
            }
            Verdict::FastError(e) => println!("      fastokens error: {e}"),
            Verdict::Same => println!("      (shrinking lost the mismatch)"),
        }
        let _ = &self.label;
    }
}

// ── Corpus ──────────────────────────────────────────────────────────────────

const QWEN_SPECIALS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];
const GEMMA_SPECIALS: &[&str] = &[
    "<bos>",
    "<|turn>",
    "<turn|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
    "<|channel>",
    "<channel|>",
];

fn corpus(family: Family, repo: &RepoText) -> Vec<(&'static str, Vec<String>)> {
    let specials = match family {
        Family::Qwen => QWEN_SPECIALS,
        Family::Gemma => GEMMA_SPECIALS,
    };
    vec![
        (
            "agentic prompts",
            (0..6)
                .map(|i| agentic(family, repo, 6_000 + i * 15_000, i))
                .collect(),
        ),
        ("multilingual + marks", multilingual()),
        ("whitespace + punctuation", whitespace_runs()),
        ("long match @ chunk edge", chunk_edge_texts(repo, None)),
        (
            "long match @ edge in turn",
            chunk_edge_texts(repo, Some(family)),
        ),
        (
            "special tokens in text",
            special_token_texts(specials, repo),
        ),
    ]
}

/// A rendered multi-turn agentic conversation of roughly `target_bytes`.
fn agentic(family: Family, repo: &RepoText, target_bytes: usize, seed: usize) -> String {
    let tools = tool_schemas(repo);
    let defs: Vec<ToolDef<'_>> = tools
        .iter()
        .map(|(name, desc, params)| ToolDef {
            name,
            description: Some(desc),
            parameters: Some(params),
            response: None,
        })
        .collect();
    let mut s = match family {
        Family::Qwen => lumen_mlx::render_tools_system_block(&defs, Some(&repo.system), None),
        Family::Gemma => {
            let mut s = format!("<bos><|turn>system\n{}", repo.system);
            for (name, desc, params) in &tools {
                s.push_str(&format!(
                    "<|tool>declaration:{name}{{description:<|\"|>{desc}<|\"|>,parameters:{params}}}<tool|>"
                ));
            }
            s.push_str("<turn|>\n");
            s
        }
    };
    let mut i = seed;
    while s.len() < target_bytes {
        let file = &repo.sources[i % repo.sources.len()];
        let ask = ASKS[i % ASKS.len()];
        match family {
            Family::Qwen => {
                s.push_str(&format!(
                    "<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\nThe user wants {ask} I should read {path} first.\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{path}\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n",
                    path = file.0,
                    body = file.1
                ));
            }
            Family::Gemma => {
                s.push_str(&format!(
                    "<|turn>user\n{ask}<turn|>\n<|turn>model\n<|tool_call>call:read_file{{path:<|\"|>{path}<|\"|>}}<tool_call|><|tool_response>response:read_file{{content:<|\"|>{body}<|\"|>}}<tool_response|>",
                    path = file.0,
                    body = file.1
                ));
            }
        }
        i += 1;
    }
    s.push_str(match family {
        Family::Qwen => "<|im_start|>assistant\n",
        Family::Gemma => "<|turn>model\n",
    });
    s
}

const ASKS: &[&str] = &[
    "can you explain how the prefix cache decides a hit?",
    "이 함수가 왜 두 번 토큰화하는지 설명해줘.",
    "リファクタリングの方針を教えてください。",
    "Fix the failing test — it's 3 lines, I think.",
    "請幫我檢查這段程式碼有沒有記憶體洩漏。",
    "why does `cargo xtask gate` take 2h?\r\nAlso: CRLF here.",
];

fn tool_schemas(repo: &RepoText) -> Vec<(String, String, Value)> {
    const NAMES: &[&str] = &[
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep_search",
        "glob",
        "run_shell",
        "web_fetch",
        "web_search",
        "todo_write",
        "ask_user",
        "git_diff",
        "git_log",
        "run_tests",
        "format_code",
        "lint",
        "open_url",
        "screenshot",
        "click",
        "type_text",
        "memory_read",
        "memory_write",
        "spawn_agent",
        "send_message",
        "schedule",
        "notebook_edit",
        "image_generate",
        "translate",
        "summarize",
        "calculator",
    ];
    NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let desc = repo.slice(i * 2_311, 1_800);
            let params = json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": repo.slice(i * 997 + 50_000, 160)},
                    "limit": {"type": "integer", "description": "Maximum number of lines; 0 = all."},
                    "mode": {"type": "string", "enum": ["fast", "full", "diff"]},
                },
                "required": ["path"],
            });
            (name.to_string(), desc, params)
        })
        .collect()
}

fn multilingual() -> Vec<String> {
    let samples = [
        "안녕하세요, 세계! 토크나이저 패리티를 검증합니다. 한국어와 English 가 섞인 code-switching.",
        "\u{1100}\u{1161}\u{11A8} \u{1112}\u{1161}\u{11AB}\u{1100}\u{1173}\u{11AF} (decomposed jamo)",
        "こんにちは世界。トークナイザーの検証です。カタカナ、ひらがな、漢字。",
        "你好，世界。这是一个没有空格的很长的中文段落用于测试分词器的一致性没有标点也没有空格",
        "مرحبا بالعالم، هذا اختبار. مُحَمَّد — النصوص من اليمين إلى اليسار",
        "नमस्ते दुनिया — क्षत्रिय, ज्ञान, श्री (viramas and matras)",
        "สวัสดีชาวโลก ภาษาไทยไม่มีช่องว่างระหว่างคำ",
        "👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽 🇰🇷🇯🇵 ❤️‍🔥 🫠🫨 🧑🏿‍🚀",
        "e\u{301} vs é, A\u{30A} vs Å vs \u{212B}, ﬁ ﬂ, ①②, Ⅻ, ½, ｆｕｌｌｗｉｄｔｈ",
        "∀x∈ℝ: x²≥0 → √(x²)=|x| ⊕ ⊗ ∮ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛",
        "a\u{200B}b\u{200C}c\u{200D}d\u{FEFF}e\u{202E}f\u{2066}g\u{00AD}h",
        "Version 3.14.159 — ٣١٤ — 三一四 — ३१४ — ①④ — 0x1F600",
        "fn main() {\n\tlet x: Vec<u8> = vec![0xFF; 1024];\r\n    println!(\"{x:?}\");\n}\n",
        "{\"key\": [1, 2.5e-3, null, true], \"nested\": {\"emoji\": \"😀\", \"esc\": \"\\u00e9\\n\"}}",
        "Ελληνικά, Русский, עברית, ქართული, Հայերեն, ᐃᓄᒃᑎᑐᑦ, ᠮᠣᠩᠭᠣᠯ",
        "'s 't 're 've 'm 'll 'd 'S 'T 'RE ’s it's they're I'M",
    ];
    let mut v: Vec<String> = samples.iter().map(|s| s.to_string()).collect();
    v.push(samples.join("\n"));
    v.push(samples.concat());
    v
}

fn whitespace_runs() -> Vec<String> {
    let mut v = Vec::new();
    for n in [1, 2, 3, 7, 16, 64, 255, 1_024, 4_096] {
        for ws in [" ", "\t", "\n", "\r\n", " \n", "\u{3000}", "\u{a0}"] {
            let run = ws.repeat(n);
            v.push(format!("word{run}word"));
            v.push(format!("{run}lead"));
            v.push(format!("trail{run}"));
            v.push(format!("a{run}1{run}!{run}"));
        }
        for p in ["=", "-", "...", "!?", "#", "*", "—"] {
            v.push(format!("x {} y", p.repeat(n)));
        }
        v.push(format!("{}{}", "1".repeat(n), "a".repeat(n)));
    }
    v
}

/// Long single regex matches (> the 1 KB chunk overlap) laid exactly across
/// fastokens' parallel-chunk authority boundaries. With `framing`, the run
/// sits inside one long tool-result segment between special tokens, which
/// takes the generic path; without, the text has no special tokens and takes
/// the single-split fast path.
fn chunk_edge_texts(repo: &RepoText, framing: Option<Family>) -> Vec<String> {
    let runs: [(&str, fn(usize) -> String); 6] = [
        ("letters", |n| {
            "abcdefghij".repeat(n / 10 + 1)[..n].to_string()
        }),
        ("spaces", |n| " ".repeat(n)),
        ("newlines", |n| "\n".repeat(n)),
        ("equals", |n| "=".repeat(n)),
        ("cjk", |n| {
            "漢字仮名交じり文"
                .repeat(n / 24 + 1)
                .chars()
                .take(n / 3)
                .collect()
        }),
        ("thai", |n| {
            "สวัสดีชาวโลก"
                .repeat(n / 36 + 1)
                .chars()
                .take(n / 3)
                .collect()
        }),
    ];
    let mut v = Vec::new();
    for seg_len in [20_000usize, 41_000, 120_000] {
        for &(_, make) in &runs {
            for run_len in [1_100, 3_000] {
                for back in [1, 512, 1_023, 1_024, 1_025] {
                    let mut seg = String::new();
                    for b in authority_boundaries(seg_len) {
                        let start = b.saturating_sub(back);
                        seg.push_str(&repo.ascii(start.saturating_sub(seg.len())));
                        seg.push_str(&make(run_len));
                    }
                    seg.push_str(&repo.ascii(seg_len.saturating_sub(seg.len())));
                    v.push(match framing {
                        None => seg,
                        Some(Family::Qwen) => format!(
                            "<|im_start|>user\n<tool_response>\n{seg}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                        ),
                        Some(Family::Gemma) => format!(
                            "<|turn>model\n<|tool_response>response:read_file{{content:<|\"|>{seg}<|\"|>}}<tool_response|><turn|>\n"
                        ),
                    });
                }
            }
        }
    }
    v
}

/// fastokens' split points for a segment of `len` bytes on this machine:
/// `n = max(2, min(cores, len / 8 KiB))` zones of `len / n` (before snapping
/// to a char boundary, which ASCII filler makes a no-op).
fn authority_boundaries(len: usize) -> Vec<usize> {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let n = cores.min(len / 8_192).max(2);
    (1..n).map(|i| i * (len / n)).collect()
}

fn special_token_texts(specials: &[&str], repo: &RepoText) -> Vec<String> {
    let mut v = Vec::new();
    for sp in specials {
        let half = sp.len() / 2;
        v.push(format!("before {sp} after"));
        v.push(format!("{sp}{sp}{sp}"));
        v.push(format!("x{sp}y"));
        v.push(format!("partial {} and {}", &sp[..half], &sp[half..]));
        v.push(format!("<{sp}>"));
        v.push(format!("{}{sp}{}", repo.ascii(9_000), repo.ascii(9_000)));
    }
    v.push(specials.concat());
    v
}

fn random_text(rng: &mut Rng, len: usize) -> String {
    const PIECES: &[&str] = &[
        "the",
        " quick",
        "brown",
        " fox",
        "'s",
        "'LL",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "\n\n",
        "0",
        "42",
        "3.14",
        "=",
        "==",
        "->",
        "{",
        "}",
        "\"",
        "\\",
        "/",
        "//",
        "#",
        "!",
        "?",
        "한국어",
        "가",
        "\u{1100}",
        "\u{1161}",
        "日本",
        "語",
        "漢字",
        "😀",
        "👍🏽",
        "\u{200D}",
        "\u{FE0F}",
        "\u{301}",
        "\u{308}",
        "é",
        "e",
        "Å",
        "\u{212B}",
        "ไทย",
        "\u{E31}",
        "क्ष",
        "\u{94D}",
        "مرحبا",
        "\u{64E}",
        "\u{3000}",
        "\u{a0}",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "<|im_",
        "end|>",
        "<|turn>",
        "<turn|>",
        "<|\"|>",
        "<bos>",
        "<",
        "|",
        ">",
        "ﬁ",
        "①",
        "𝔘",
        "\u{0}",
        "\u{7f}",
        "\u{FFFD}",
    ];
    let mut s = String::with_capacity(len + 16);
    while s.len() < len {
        s.push_str(PIECES[rng.below(PIECES.len())]);
    }
    s
}

fn split_at_specials<'a>(text: &'a str, specials: &[&str]) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    'outer: while !rest.is_empty() {
        let mut best: Option<(usize, usize)> = None;
        for sp in specials {
            if let Some(i) = rest.find(sp) {
                if best.is_none_or(|(b, _)| i < b) {
                    best = Some((i, sp.len()));
                }
            }
        }
        match best {
            Some((i, n)) => {
                if i > 0 {
                    pieces.push(&rest[..i]);
                }
                rest = &rest[i + n..];
            }
            None => {
                pieces.push(rest);
                break 'outer;
            }
        }
    }
    pieces
}

// ── Repo text ───────────────────────────────────────────────────────────────

/// Realistic text from this repository: docs for system prompts and tool
/// descriptions, source files for tool results.
struct RepoText {
    system: String,
    docs: String,
    ascii: String,
    sources: Vec<(String, String)>,
}

impl RepoText {
    fn load() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut docs = String::new();
        for dir in ["docs", "."] {
            if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
                let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                files.sort();
                for p in files
                    .iter()
                    .filter(|p| p.extension().is_some_and(|x| x == "md"))
                {
                    docs.push_str(&std::fs::read_to_string(p).unwrap_or_default());
                }
            }
        }
        let mut sources = Vec::new();
        for dir in [
            "crates/lumen-server/src",
            "crates/lumen-mlx/src",
            "crates/lumen-core/src",
        ] {
            if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
                let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                files.sort();
                for p in files
                    .iter()
                    .filter(|p| p.extension().is_some_and(|x| x == "rs"))
                {
                    let body = std::fs::read_to_string(p).unwrap_or_default();
                    let cut = floor_char(&body, 12_000);
                    sources.push((
                        format!("{dir}/{}", p.file_name().unwrap().to_string_lossy()),
                        body[..cut].to_string(),
                    ));
                }
            }
        }
        assert!(docs.len() > 100_000, "repo docs not found from {root:?}");
        assert!(!sources.is_empty(), "repo sources not found from {root:?}");
        let ascii: String = docs.chars().filter(|c| c.is_ascii()).collect();
        let cut = floor_char(&docs, 20_000);
        Self {
            system: docs[..cut].to_string(),
            docs,
            ascii,
            sources,
        }
    }

    /// `len` chars of the docs starting at `at` (wrapping).
    fn slice(&self, at: usize, len: usize) -> String {
        self.docs
            .chars()
            .cycle()
            .skip(at % self.docs.len())
            .take(len)
            .collect()
    }

    /// Exactly `len` bytes of ASCII text, so byte offsets land where asked.
    fn ascii(&self, len: usize) -> String {
        self.ascii
            .bytes()
            .cycle()
            .take(len)
            .map(char::from)
            .collect()
    }
}

// ── Small helpers ───────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

fn median_ms(mut f: impl FnMut(usize)) -> f64 {
    f(usize::MAX);
    let mut runs: Vec<f64> = (0..7)
        .map(|run| {
            let t0 = Instant::now();
            f(run);
            ms(t0)
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[runs.len() / 2]
}
