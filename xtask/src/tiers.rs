//! `cargo xtask tiers` — what the 165 ignored tests are waiting for
//! (005 Phase 5).
//!
//! A skipped test is indistinguishable from a passing one in the summary line,
//! and that is the whole problem: `cargo xtask test` reports **152 ignored**
//! and nothing about the run says whether that is 152 tests needing a GPU we
//! have, or 152 tests nobody can run at all. The `causal-mask-coverage` defect
//! in `red_green.rs` was exactly this — an `#[ignore]`d test panicking in
//! silence, for long enough that the panic became the status quo.
//!
//! Rather than re-annotating 165 attributes with a tier enum, this reads the
//! **reason string each `#[ignore]` already carries** and classifies it. The
//! tiers were never missing; they were just not machine-readable.
//!
//! The enforceable half is [`bare_ignores`]: an `#[ignore]` with no reason at
//! all cannot be classified, cannot be scheduled, and cannot be decided about.
//! Those are reported as an error, because they are the ones that quietly
//! become permanent.

use std::collections::BTreeMap;
use std::process::{Command, ExitCode};

/// What a tier needs before its tests can run. Matched against the `#[ignore]`
/// reason text, first match wins, so the more specific patterns come first.
struct Tier {
    id: u8,
    name: &'static str,
    needs: &'static str,
    /// Lowercase substrings that identify this tier in a reason string.
    markers: &'static [&'static str],
}

const TIERS: &[Tier] = &[
    // First, because it is the only tier nobody can satisfy. These tests need
    // reference dumps that no committed script regenerates, so classifying them
    // by the *other* thing they need — a Metal device — would say "runnable on
    // this machine" about a test that is runnable nowhere.
    Tier {
        id: 4,
        name: "unreproducible reference",
        needs: "dev-session dumps that no committed script produces; see \
                crates/lumen-mlx/tests/golden/README.md for the two honest ways out",
        markers: &["no committed script produces"],
    },
    Tier {
        id: 3,
        name: "soak / external process",
        needs: "a Python environment and a multi-GB download; minutes to hours",
        markers: &["spawns python", "downloads/loads", "swift dump"],
    },
    Tier {
        id: 2,
        name: "GPU + full weights",
        needs: "a Metal device AND the model shards (~16 GB) on local disk",
        // "model dir" and "checkpoint" are how most of this repo phrases
        // "needs the weights"; without them 16 tests read as unclassified and
        // the report loses its point.
        markers: &[
            "16 gb",
            "16gb",
            "lmstudio shards",
            "model_dir",
            "model dir",
            "checkpoint",
            "full mlx model",
            "captured fixture",
        ],
    },
    Tier {
        id: 2,
        name: "small fixture",
        needs: "a tokenizer.json extracted from a checkpoint (~5 MB)",
        markers: &["tokenizer.json", "tokenizer"],
    },
    Tier {
        id: 1,
        name: "GPU, synthetic tensors",
        needs: "a Metal device — no weights, no download",
        // "mlx-sys" catches the raw-FFI diagnostic smokes, which say what they
        // exercise but never say "Metal" — they need a device all the same.
        markers: &["metal", "mlx ffi", "mlx-sys", "mlx-rs", "non-sandbox"],
    },
];

pub fn main(_args: Vec<String>) -> ExitCode {
    let entries = match collect() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("could not scan for #[ignore]: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut by_tier: BTreeMap<(u8, &str), Vec<&Entry>> = BTreeMap::new();
    let mut unclassified: Vec<&Entry> = Vec::new();
    for e in &entries {
        match classify(&e.reason) {
            Some(t) => by_tier.entry((t.id, t.name)).or_default().push(e),
            None => unclassified.push(e),
        }
    }

    println!("=== ignored tests by tier ===\n");
    println!("  tier 0 runs on every `cargo xtask test`; everything below is skipped there.\n");
    for ((id, name), items) in &by_tier {
        let needs = TIERS
            .iter()
            .find(|t| t.id == *id && t.name == *name)
            .map(|t| t.needs)
            .unwrap_or("");
        println!("  tier {id} — {name}: {} tests", items.len());
        println!("      needs: {needs}");
    }

    if !unclassified.is_empty() {
        println!("\n  unclassified: {} tests", unclassified.len());
        println!("      the reason string matches no known tier — either the reason is");
        println!("      too vague to schedule against, or a tier is missing here.");
        for e in unclassified.iter().take(8) {
            println!(
                "      {}:{} — {:?}",
                e.file,
                e.line,
                truncate(&e.reason, 60)
            );
        }
        if unclassified.len() > 8 {
            println!("      … and {} more", unclassified.len() - 8);
        }
    }

    let undeclared = match collect_external_deps() {
        Ok(deps) => deps
            .into_iter()
            .filter(|d| !d.reason.as_deref().is_some_and(declares_external_input))
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("could not scan for external inputs: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("\n=== undeclared external inputs ===\n");
    if undeclared.is_empty() {
        println!("  none — every test reading a path outside the repo says so in its reason");
    } else {
        println!(
            "  {} tests read a path outside the repo without saying so.\n",
            undeclared.len()
        );
        for d in &undeclared {
            let state = match &d.reason {
                None => "NOT ignored — fails on every machine without the file".to_string(),
                Some(r) => format!("ignore reason: {:?}", truncate(r, 64)),
            };
            println!("  {}:{} {}", d.file, d.line, d.test);
            println!("      reads: {}", d.paths.join(", "));
            println!("      {state}");
        }
        println!(
            "\n  A reason can be present, classifiable, and still wrong about what the\n\
             \x20 test needs. `flux-scheduler-invariants` in red_green.rs was exactly this\n\
             \x20 — a comparison against /tmp/klein_sigmas.bin, a dev-session dump that had\n\
             \x20 ceased to exist, so the test failed on every machine but one.\n\n\
             \x20 Say it where the reader will look:\n\
             \x20   #[ignore = \"needs a Metal device AND /tmp/klein_*.bin reference dumps;\n\
             \x20                no committed script produces them — see\n\
             \x20                crates/lumen-mlx/tests/golden/README.md\"]\n\n\
             \x20 The dump does not have to exist. Some references genuinely cannot be\n\
             \x20 regenerated. The requirement just has to stop being invisible."
        );
    }

    let bare = bare_ignores(&entries);
    println!("\n=== reasonless #[ignore] ===\n");
    if bare.is_empty() {
        println!("  none — every skipped test says what it is waiting for");
        println!("\ntotal ignored: {}", entries.len());
        if undeclared.is_empty() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    } else {
        println!("  {} tests are `#[ignore]`d with NO reason.\n", bare.len());
        for e in &bare {
            println!("  {}:{}", e.file, e.line);
        }
        println!(
            "\n  An ignore with no reason cannot be classified, scheduled, or decided\n\
             \x20 about — it is the shape that quietly becomes permanent. Give each one\n\
             \x20 a reason: `#[ignore = \"needs a Metal device\"]`.\n\n\
             total ignored: {}",
            entries.len()
        );
        ExitCode::FAILURE
    }
}

struct Entry {
    file: String,
    line: usize,
    /// Empty for a bare `#[ignore]`.
    reason: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Undeclared external inputs
// ─────────────────────────────────────────────────────────────────────────────
//
// A reason string can be present, well-formed, classifiable — and still wrong
// about what the test needs. Seven tests in `lumen-diffusion` said
// `"MLX FFI requires non-sandbox host with Metal device"` while also reading
// `/tmp/klein_image.bin`, `/tmp/dit_out.bin` and friends: dev-session dumps that
// no committed script produces. Someone with a Metal device runs `--ignored`,
// gets a failure, and the reason string gives them nothing to go on.
//
// This is the `flux-scheduler-invariants` defect in `red_green.rs` — a test
// comparing against `/tmp/klein_sigmas.bin`, a dump that had ceased to exist —
// except that one was found in production and these seven were not. Fixing the
// one file it was found in left the pattern in place everywhere else.
//
// The rule: a test that reads a hardcoded path outside the repo has to say so
// where the reader will look, which is the `#[ignore]` reason. It does not
// require the dump to exist — some references genuinely cannot be regenerated
// (see `crates/lumen-mlx/tests/golden/README.md`) — only that the requirement
// stop being invisible.

/// Absolute-path prefixes that are outside the repo by construction. A test
/// reading one of these depends on something no checkout provides.
const EXTERNAL_PREFIXES: &[&str] = &["/tmp/", "/Users/", "/var/folders/", "/private/tmp/"];

/// Files exempt from the external-input rule because they **define** it. Same
/// reasoning as `RULE_DEFINITIONS` in `gate.rs`: a rule that fails on its own
/// definition is a rule people learn to route around.
const EXTERNAL_RULE_DEFINITIONS: &[&str] = &["xtask/src/tiers.rs"];

/// A file's lines with multi-line attributes joined into one.
///
/// Both scanners here originally read one physical line at a time, which was
/// fine until a reason string got long enough to wrap. Then `#[ignore = "…\`
/// spanned three lines, `attrs_above` hit a continuation line that was neither
/// an attribute nor a comment, stopped, and concluded the function had no
/// `#[test]` at all — so seven tests dropped out of the scan and the rule
/// reported "none" for the exact files it had just been written to catch.
///
/// Reporting zero because you stopped looking is the failure this command
/// exists to prevent, so the fix belongs in the scanner and not in the reasons.
fn logical_lines(src: &str) -> Vec<(usize, String)> {
    let raw: Vec<&str> = src.lines().collect();
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let start = i;
        let mut joined = raw[i].trim().to_string();
        if joined.starts_with("#[") {
            // Keep absorbing lines until the attribute closes. A wrapped string
            // continuation ends in `\`; the last line ends in `]`.
            while !attr_is_closed(&joined) && i + 1 < raw.len() {
                i += 1;
                joined = format!(
                    "{} {}",
                    joined.trim_end_matches('\\').trim_end(),
                    raw[i].trim()
                );
            }
        }
        out.push((start + 1, joined));
        i += 1;
    }
    out
}

/// Has this accumulated attribute text reached its closing bracket?
fn attr_is_closed(text: &str) -> bool {
    let code = match text.find("//") {
        Some(k) => &text[..k],
        None => text,
    };
    code.trim_end().ends_with(']')
}

/// A `#[test]` that names an external path, and what its `#[ignore]` says.
struct ExternalDep {
    file: String,
    line: usize,
    test: String,
    /// `None` when the test is not ignored at all — worse, since it then runs
    /// and fails on every machine that lacks the file.
    reason: Option<String>,
    paths: Vec<String>,
}

/// Does this reason tell the reader about an external input?
///
/// Either it names the path directly, or it names the env var that overrides it
/// — `lumen-diffusion`'s tokenizer test already did the latter
/// (`"needs a tokenizer and a Swift dump; set LUMEN_DIFFUSION_SWIFT_TOKENS"`),
/// which is the shape the other seven should have had.
fn declares_external_input(reason: &str) -> bool {
    reason.contains("/tmp")
        || reason.contains("/Users")
        || reason.contains("dump")
        || reason
            .split(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .any(|w| w.starts_with("LUMEN_") && w.len() > "LUMEN_".len())
}

/// Every `#[test]` whose body names a path outside the repo.
///
/// Deliberately a line scanner rather than a parse: it has to run in the gate,
/// and the thing it is looking for — a string literal starting `/tmp/` — is not
/// ambiguous at the token level.
fn collect_external_deps() -> std::io::Result<Vec<ExternalDep>> {
    let out = Command::new("git")
        .args(["ls-files", "crates/*.rs", "xtask/*.rs"])
        .output()?;
    let mut found = Vec::new();
    for file in String::from_utf8_lossy(&out.stdout).lines() {
        if EXTERNAL_RULE_DEFINITIONS.iter().any(|d| file.ends_with(d)) {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        let logical = logical_lines(&src);
        let lines: Vec<&str> = logical.iter().map(|(_, s)| s.as_str()).collect();
        for (i, l) in lines.iter().enumerate() {
            let Some(name) = fn_name(l) else { continue };
            let attrs = attrs_above(&lines, i);
            if !attrs.iter().any(|a| a.starts_with("#[test]")) {
                continue;
            }
            let paths = external_paths_in_body(&lines, i);
            if paths.is_empty() {
                continue;
            }
            let reason = attrs.iter().find(|a| a.starts_with("#[ignore")).map(|a| {
                a.split_once('"')
                    .and_then(|(_, r)| r.split('"').next())
                    .unwrap_or("")
                    .to_string()
            });
            found.push(ExternalDep {
                file: file.to_string(),
                line: logical[i].0,
                test: name,
                reason,
                paths,
            });
        }
    }
    Ok(found)
}

fn fn_name(line: &str) -> Option<String> {
    let t = line.trim_start();
    let rest = t.strip_prefix("fn ").or_else(|| {
        t.strip_prefix("pub fn ")
            .or_else(|| t.strip_prefix("async fn "))
    })?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty() && rest[name.len()..].starts_with('(')).then_some(name)
}

/// The attribute lines immediately above `i`, nearest first. Comments and blank
/// lines are skipped rather than terminating the scan — a doc comment between
/// `#[test]` and `fn` is normal here.
fn attrs_above(lines: &[&str], i: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut j = i;
    while j > 0 {
        j -= 1;
        let t = lines[j].trim();
        if t.starts_with("#[") {
            out.push(t.to_string());
        } else if t.is_empty() || t.starts_with("//") {
            continue;
        } else {
            break;
        }
    }
    out
}

/// External path literals inside the function body starting at `i`, found by
/// brace counting from the signature line.
fn external_paths_in_body(lines: &[&str], i: usize) -> Vec<String> {
    let mut depth = 0i32;
    let mut started = false;
    let mut paths = Vec::new();
    for l in &lines[i..] {
        for c in l.chars() {
            if c == '{' {
                depth += 1;
                started = true;
            } else if c == '}' {
                depth -= 1;
            }
        }
        for lit in string_literals(l) {
            if EXTERNAL_PREFIXES.iter().any(|p| lit.starts_with(p)) && !paths.contains(&lit) {
                paths.push(lit);
            }
        }
        if started && depth <= 0 {
            break;
        }
    }
    paths
}

fn string_literals(line: &str) -> Vec<String> {
    // Comments do not count: prose *about* `/tmp/foo.bin` in a doc comment is
    // not a dependency, and counting it would make the rule cry wolf on the
    // very notes that explain the dependency.
    let code = match line.find("//") {
        Some(k) => &line[..k],
        None => line,
    };
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    let mut prev_backslash = false;
    for c in code.chars() {
        match &mut cur {
            None => {
                if c == '"' {
                    cur = Some(String::new());
                }
            }
            Some(buf) => {
                if c == '"' && !prev_backslash {
                    out.push(std::mem::take(buf));
                    cur = None;
                } else {
                    buf.push(c);
                }
            }
        }
        prev_backslash = c == '\\' && !prev_backslash;
    }
    out
}

/// Every `#[ignore]` attribute in the workspace's crates, with its reason.
///
/// Reads whole files rather than `git grep`ping for the attribute, because a
/// wrapped reason string is several physical lines and a line-wise match
/// truncates it at the wrap — which silently drops the most specific half of
/// the sentence, and with it the tier the test actually belongs to.
fn collect() -> std::io::Result<Vec<Entry>> {
    let out = Command::new("git")
        .args(["ls-files", "crates/*.rs"])
        .output()?;
    let mut entries = Vec::new();
    for file in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        for (no, text) in logical_lines(&src) {
            // Only the attribute itself — prose *about* `#[ignore]` in a doc
            // comment is not an ignored test, and counting it would inflate the
            // number this command exists to make honest.
            if !text.starts_with("#[ignore") {
                continue;
            }
            let reason = text
                .split_once('"')
                .and_then(|(_, r)| r.split('"').next())
                .unwrap_or("")
                .to_string();
            entries.push(Entry {
                file: file.to_string(),
                line: no,
                reason,
            });
        }
    }
    Ok(entries)
}

fn classify(reason: &str) -> Option<&'static Tier> {
    if reason.trim().is_empty() {
        return None;
    }
    let lower = reason.to_lowercase();
    TIERS
        .iter()
        .find(|t| t.markers.iter().any(|m| lower.contains(m)))
}

fn bare_ignores(entries: &[Entry]) -> Vec<&Entry> {
    entries
        .iter()
        .filter(|e| e.reason.trim().is_empty())
        .collect()
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().take(n).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classifier must actually classify the reasons this repo uses. A tier
    /// table that matches nothing would report every test as unclassified and
    /// teach the reader to ignore the output.
    #[test]
    fn the_real_reason_strings_classify() {
        let cases = [
            ("MLX FFI requires non-sandbox host with Metal device", 1),
            ("requires tokenizer.json from lmstudio shards (~5 MB)", 2),
            ("requires lmstudio shards (~16 GB) + Metal", 2),
            ("spawns Python + downloads/loads ~19GB model", 3),
            ("requires local Gemma 4 model dir; run with --ignored", 2),
            ("loads full MLX model; set LUMEN_MLX_GOLDEN_IN", 2),
            (
                "diagnostic smoke for the raw mlx-sys int allocation path",
                1,
            ),
        ];
        for (reason, want) in cases {
            let t = classify(reason).unwrap_or_else(|| panic!("unclassified: {reason:?}"));
            assert_eq!(t.id, want, "{reason:?} landed in tier {}", t.id);
        }
    }

    /// Specific before general: a reason naming both weights and Metal is
    /// tier 2, not tier 1, or the schedule under-states what it needs.
    #[test]
    fn the_more_specific_tier_wins() {
        let both =
            "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)";
        assert_eq!(classify(both).expect("classified").id, 2);
    }

    /// A bare ignore is unclassifiable by construction, and that is the point.
    #[test]
    fn a_reasonless_ignore_is_not_classified() {
        assert!(classify("").is_none());
        assert!(classify("   ").is_none());
    }

    /// A test needing a dump nobody can produce must not classify as "just
    /// needs a Metal device". That was the whole failure: seven tests said
    /// Metal, meant Metal *and* seven files that do not exist, and the report
    /// repeated the understatement back.
    #[test]
    fn an_unreproducible_reference_outranks_the_metal_marker() {
        let reason = "needs a Metal device AND /tmp/vae_{latent,out}.bin reference \
                      dumps; no committed script produces them";
        let t = classify(reason).expect("classified");
        assert_eq!(
            t.id, 4,
            "landed in tier {} — a test nobody can run must not read as tier 1",
            t.id
        );
    }

    /// The declaration check is the enforceable half, so its predicate gets
    /// pinned on both sides. The `LUMEN_*` case is the shape `tokenizer.rs`
    /// already used correctly and the other seven did not.
    #[test]
    fn a_reason_declares_an_external_input_only_when_it_says_so() {
        assert!(declares_external_input("needs /tmp/klein_image.bin"));
        assert!(declares_external_input(
            "needs a tokenizer and a Swift dump; set LUMEN_DIFFUSION_SWIFT_TOKENS"
        ));
        assert!(!declares_external_input(
            "MLX FFI requires non-sandbox host with Metal device"
        ));
        assert!(
            !declares_external_input("needs LUMEN_"),
            "a bare prefix names no variable"
        );
    }

    /// A wrapped reason must survive the scan intact. This is the bug that made
    /// the rule report "none" for the seven files it was written to catch: the
    /// scanner read one physical line, so `"…produces them — see \` lost its
    /// tail, and with it the tier-4 marker.
    #[test]
    fn a_wrapped_attribute_is_read_as_one_attribute() {
        let src = "    #[test]\n    \
                   #[ignore = \"needs a Metal device AND /tmp/vae_out.bin; \\\n                \
                   no committed script produces them\"]\n    fn t() {}\n";
        let logical = logical_lines(src);
        let attr = logical
            .iter()
            .map(|(_, s)| s)
            .find(|s| s.starts_with("#[ignore"))
            .expect("the ignore attribute survived joining");
        assert!(
            attr.contains("no committed script produces them"),
            "the tail was truncated: {attr:?}"
        );
        let reason = attr.split_once('"').unwrap().1.split('"').next().unwrap();
        assert_eq!(
            classify(reason).expect("classified").id,
            4,
            "a truncated reason silently demotes the test to tier 1"
        );
    }

    /// The body scanner has to see past prose. A doc comment mentioning a dump
    /// is an explanation, not a dependency, and flagging it would make the rule
    /// fire on the notes written to satisfy it.
    #[test]
    fn a_path_in_a_comment_is_not_a_dependency() {
        let lines = [
            "    fn t() {",
            "        // reads /tmp/foo.bin when the dump exists",
            "        let x = 1;",
            "    }",
        ];
        assert!(external_paths_in_body(&lines, 0).is_empty());

        let real = ["    fn t() {", "        read(\"/tmp/foo.bin\");", "    }"];
        assert_eq!(external_paths_in_body(&real, 0), vec!["/tmp/foo.bin"]);
    }

    /// Every tier states what it needs. A tier whose `needs` is empty tells the
    /// reader a count and nothing actionable.
    #[test]
    fn every_tier_states_what_it_needs() {
        for t in TIERS {
            assert!(t.needs.len() > 20, "tier {} needs a real description", t.id);
            assert!(!t.markers.is_empty(), "tier {} matches nothing", t.id);
        }
    }
}
