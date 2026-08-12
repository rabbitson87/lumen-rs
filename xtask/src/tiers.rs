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

    let bare = bare_ignores(&entries);
    println!("\n=== reasonless #[ignore] ===\n");
    if bare.is_empty() {
        println!("  none — every skipped test says what it is waiting for");
        println!("\ntotal ignored: {}", entries.len());
        ExitCode::SUCCESS
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

/// Every `#[ignore]` attribute in the workspace's crates, with its reason.
fn collect() -> std::io::Result<Vec<Entry>> {
    let out = Command::new("git")
        .args(["grep", "-n", "--", "#\\[ignore", "crates/"])
        .output()?;
    let mut entries = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(file), Some(no), Some(text)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let Ok(no) = no.parse::<usize>() else {
            continue;
        };
        let t = text.trim();
        // Only the attribute itself — prose *about* `#[ignore]` in a doc
        // comment is not an ignored test, and counting it would inflate the
        // number this command exists to make honest.
        if !t.starts_with("#[ignore") {
            continue;
        }
        let reason = t
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
