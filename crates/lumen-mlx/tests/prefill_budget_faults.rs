//! Allocation-pressure sweep over the prefill chunk guard (005 Phase 3).
//!
//! This is the port of SQLite's `malloc()` fail-at-N testing, and the
//! translation is worth stating because it is not a literal one. SQLite can
//! return `SQLITE_NOMEM` because its allocator is fallible; lumen cannot —
//! MLX allocation returns no error to Rust, and on Apple unified memory an
//! oversized request swaps or aborts inside C++ rather than failing. So the
//! defense is not to survive the failed allocation but to **never issue it**:
//! the prefill chunk is clamped so one chunk's `[heads, q_len, kv_len]` scores
//! buffer stays under a byte budget.
//!
//! That moves the entire OOM class into pure integer arithmetic, and the sweep
//! shape carries over unchanged: rather than failing allocation 1, 2, 3, …,
//! lower the *budget* through every magnitude and require the same invariants
//! at each step:
//!
//!   1. the chunk is never 0 — the prefill loop advances by `chunk`, so a zero
//!      chunk is an infinite hang, strictly worse than an OOM;
//!   2. the chunk never exceeds what was requested — clamping is downward only,
//!      or the guard would *cause* the allocation it exists to prevent;
//!   3. the projection stays within budget except at the documented
//!      `MIN_CHUNK` floor, and there the overrun is bounded and reported.
//!
//! Property 3 is the interesting one: the floor deliberately wins over the
//! budget, because chunks that small cost more in `eval` barriers than they
//! save in memory. This sweep pins exactly where that window opens rather than
//! leaving it to be rediscovered.

use lumen_mlx::prefill_budget::{
    ChunkDecision, DEFAULT_SCORES_BUDGET_BYTES, MIN_CHUNK, clamp_chunk, parse_scores_budget,
    projected_scores_bytes,
};

/// Head counts spanning the models this serves plus the degenerate ends.
const HEADS: &[usize] = &[0, 1, 8, 16, 32, 40, 64, 128];

/// Prompt lengths: short chat, agentic system prompt, the long-context claims,
/// and an absurd one.
const PROMPTS: &[usize] = &[
    0,
    1,
    16,
    512,
    2048,
    8_192,
    32_768,
    131_072,
    262_144,
    1 << 24,
];

/// The three invariants, asserted together so every case in every sweep below
/// gets all of them.
fn assert_invariants(d: &ChunkDecision, ctx: &str) {
    assert!(
        d.chunk >= 1,
        "{ctx}: chunk 0 — the prefill loop advances by `chunk`, so this hangs forever"
    );
    assert!(
        d.chunk <= d.requested,
        "{ctx}: clamp went UP ({} > {}) — the guard would cause the allocation it prevents",
        d.chunk,
        d.requested
    );
    if !d.floored() {
        assert!(
            d.projected_bytes() <= d.budget_bytes,
            "{ctx}: {} projected bytes exceed the {} budget without the floor being \
             responsible — the guard silently does not hold",
            d.projected_bytes(),
            d.budget_bytes
        );
    }
}

/// The sweep: every budget from 1 byte upward through every magnitude, against
/// every head count and prompt length. ~3k cases, microseconds each.
#[test]
fn budget_sweep_holds_every_invariant() {
    let mut budgets: Vec<u64> = Vec::new();
    let mut b: u64 = 1;
    while b < 1 << 42 {
        // ±1 around each magnitude — where integer division actually goes wrong.
        budgets.extend([b.saturating_sub(1), b, b + 1]);
        b = b.saturating_mul(2);
    }
    budgets.push(DEFAULT_SCORES_BUDGET_BYTES);
    budgets.push(u64::MAX);
    budgets.sort_unstable();
    budgets.dedup();

    let mut floored_cases = 0usize;
    let mut total = 0usize;
    for &budget in &budgets {
        for &heads in HEADS {
            for &n in PROMPTS {
                for &requested in &[1usize, 256, 2048, 4096, usize::MAX] {
                    let d = clamp_chunk(requested, budget, heads, n);
                    assert_invariants(
                        &d,
                        &format!("budget={budget} heads={heads} n={n} requested={requested}"),
                    );
                    total += 1;
                    if d.floored() {
                        floored_cases += 1;
                    }
                }
            }
        }
    }
    lumen_testkit::cases(total, "prefill budget sweep");
    assert!(total > 2_000, "sweep should be dense, got {total} cases");
    assert!(
        floored_cases > 0,
        "no case hit the MIN_CHUNK floor — the sweep never reached the regime this \
         test exists to pin"
    );
}

/// The floor's budget-overrun window, stated numerically.
///
/// A guard that promises "scores stay under 8 GB" and then knowingly exceeds it
/// is only acceptable if the overrun is bounded and understood. This computes
/// the exact prompt length at which the window opens for a realistic model and
/// how far over it goes at the longest advertised context, so the trade is a
/// recorded number rather than a comment.
#[test]
fn the_floor_overruns_the_budget_only_beyond_realistic_context() {
    let heads = 40usize; // upper end of what this serves
    let budget = DEFAULT_SCORES_BUDGET_BYTES;

    // Smallest prompt at which the budget alone would ask for < MIN_CHUNK.
    let mut opens_at = None;
    for n in (1_000..1_000_000).step_by(1_000) {
        if clamp_chunk(4096, budget, heads, n).floored() {
            opens_at = Some(n);
            break;
        }
    }
    let opens_at = opens_at.expect("the floor window must open somewhere below 1M tokens");
    assert!(
        opens_at > 100_000,
        "the floor overruns the budget at only {opens_at} tokens — that is inside the \
         context lengths this serves, so the guard does not hold where it matters"
    );

    // At 256K context (the longest advertised) the overrun must still fit a
    // Metal buffer on the boxes this targets (~16 GB+ maxBufferLength).
    let d = clamp_chunk(4096, budget, heads, 262_144);
    assert!(d.floored(), "256K context should be in the floored regime");
    assert_eq!(d.chunk, MIN_CHUNK);
    let projected = d.projected_bytes();
    assert!(
        projected < 16_000_000_000,
        "at 256K context the floored chunk projects {projected} bytes, past the Metal \
         per-buffer cap — the floor is no longer a latency trade, it is an OOM"
    );
    assert!(
        d.over_budget_bytes() > 0,
        "this case is supposed to be over budget; if it is not, the floor analysis moved"
    );
}

/// Short prompts must be untouched. A guard that clamps the common path is a
/// throughput regression disguised as safety.
#[test]
fn ordinary_prompts_are_not_clamped() {
    for &heads in &[16usize, 32, 40] {
        for &n in &[16usize, 512, 2048, 8_192] {
            let d = clamp_chunk(2048, DEFAULT_SCORES_BUDGET_BYTES, heads, n);
            assert_eq!(
                d.chunk, 2048,
                "heads={heads} n={n} was clamped to {} — the default budget must leave \
                 ordinary prompts alone",
                d.chunk
            );
            assert!(!d.clamped());
        }
    }
}

/// Monotonicity: a longer prompt, or more heads, can only shrink the chunk.
/// A non-monotone guard would let a *larger* allocation through at a larger
/// input, which is the failure this whole mechanism is supposed to exclude.
#[test]
fn the_guard_is_monotone_in_prompt_length_and_heads() {
    for &heads in HEADS {
        let mut prev = usize::MAX;
        for &n in PROMPTS {
            let c = clamp_chunk(4096, DEFAULT_SCORES_BUDGET_BYTES, heads, n).chunk;
            assert!(
                c <= prev,
                "heads={heads}: chunk grew from {prev} to {c} as the prompt got longer"
            );
            prev = c;
        }
    }
    for &n in &[2048usize, 32_768, 262_144] {
        let mut prev = usize::MAX;
        for &heads in HEADS {
            let c = clamp_chunk(4096, DEFAULT_SCORES_BUDGET_BYTES, heads, n).chunk;
            assert!(
                c <= prev,
                "n={n}: chunk grew from {prev} to {c} as head count rose"
            );
            prev = c;
        }
    }
}

/// A hostile env value must never produce a chunk of 0 (hang) or a budget of 0
/// (everything floored). This is the fault-injection half: the budget comes
/// from an environment variable a user edits.
#[test]
fn hostile_budget_env_values_never_hang_or_disable_the_guard() {
    let hostile = [
        "",
        "0",
        "-0",
        "-1",
        "-1e9",
        "NaN",
        "nan",
        "inf",
        "-inf",
        "abc",
        "1e-300",
        "1e300",
        "0.0000001",
        " 8",
        "8 ",
        "8GB",
        "٨",
    ];
    for raw in hostile {
        let budget = parse_scores_budget(Some(raw));
        assert!(budget > 0, "{raw:?} produced a zero budget");
        let d = clamp_chunk(2048, budget, 40, 32_768);
        assert_invariants(&d, &format!("env={raw:?}"));
        assert!(d.chunk >= 1);
    }
    assert_eq!(
        parse_scores_budget(Some("1e-300")),
        DEFAULT_SCORES_BUDGET_BYTES,
        "a budget that rounds to zero bytes must fall back, not floor everything"
    );
}

/// The projection helper must saturate rather than wrap — a wrapped product
/// reads as "small enough" and waves the allocation through, which is the one
/// arithmetic bug in this file that would be silent in production.
#[test]
fn projection_never_wraps_into_looking_safe() {
    let huge = projected_scores_bytes(u64::MAX, u64::MAX, u64::MAX);
    assert_eq!(huge, u64::MAX);
    let d = clamp_chunk(usize::MAX, u64::MAX, usize::MAX, usize::MAX);
    assert_invariants(&d, "all-max");
}
