//! The client meta-wrapper heuristic, both directions (005 Phase 4.1).
//!
//! `is_client_meta_wrapper` decides whether to **silently delete a user
//! message**. Both of its failure modes are invisible: a false positive drops
//! something the user actually wrote, and a false negative leaves the wrapper
//! in the prompt, which is the drift this heuristic exists to prevent
//! (repetition loops and `max_tokens` runaway from an agent client re-injecting
//! "if you have completed the task, call task_complete" every turn).
//!
//! The predicate is `opener && signal` over four openers and four signals, and
//! coverage said exactly **one of each** had ever been exercised. Adding an
//! opener that never matches, or breaking one that used to, changes which
//! messages get deleted with nothing anywhere reporting it.
//!
//! Table-driven in both directions on purpose: the accept table proves the
//! heuristic still fires, and the reject table proves it has not widened into
//! deleting ordinary prose.

use lumen_mlx::chat_io::{ChatTurn, is_client_meta_wrapper, strip_client_meta_wrappers_indexed};

/// The four documented openers. Every one must be able to fire on its own.
const OPENERS: &[&str] = &[
    "If you have",
    "If the task",
    "Once you have",
    "When the task",
];

/// The four documented signals. `call the … tool` is a conjunction, so it is
/// spelled out in full here.
const SIGNALS: &[&str] = &[
    "call task_complete now.",
    "and it is fully completed, stop.",
    "give the final answer.",
    "call the completion tool.",
];

/// Compose a wrapper long enough to clear the 30-character floor.
fn wrapper(opener: &str, signal: &str) -> String {
    format!("{opener} finished the work, {signal} Thanks.")
}

#[test]
fn every_opener_fires_with_every_signal() {
    for o in OPENERS {
        for s in SIGNALS {
            let text = wrapper(o, s);
            assert!(
                is_client_meta_wrapper(&text),
                "opener {o:?} + signal {s:?} should be recognised: {text:?}"
            );
        }
    }
}

/// The predicate needs BOTH halves. This is the documented false-positive
/// guard — "if you have time, please be concise" is ordinary prose and must
/// survive.
#[test]
fn neither_half_is_sufficient_on_its_own() {
    for o in OPENERS {
        let text = format!("{o} a moment, please keep the answer short and readable.");
        assert!(
            !is_client_meta_wrapper(&text),
            "an opener with no completion signal is ordinary prose: {text:?}"
        );
    }
    for s in SIGNALS {
        let text = format!("Here is some ordinary user prose that happens to {s} Thanks.");
        assert!(
            !is_client_meta_wrapper(&text),
            "a signal with no meta-instruction opener is ordinary prose: {text:?}"
        );
    }
}

/// `call the … tool` is an AND of two substrings; each alone must not count,
/// or "call the police" and "the tool broke" start deleting messages.
#[test]
fn the_call_the_tool_signal_needs_both_of_its_substrings() {
    let both = "If you have finished, call the completion tool as instructed here.";
    assert!(is_client_meta_wrapper(both));

    for half in [
        "If you have finished, call the number listed in the document above.",
        "If you have finished, the tool you were given is no longer needed here.",
    ] {
        assert!(
            !is_client_meta_wrapper(half),
            "half of the conjunction must not fire: {half:?}"
        );
    }
}

/// The length window is `[30, 500]`, and both ends are exclusive-outside.
/// The upper bound is what keeps a long user message that merely *quotes* an
/// agent instruction from being deleted wholesale.
#[test]
fn the_length_window_is_enforced_at_both_ends() {
    let core = "If you have done it, call task_complete.";
    assert!(is_client_meta_wrapper(core), "the base case must match");
    assert!(core.len() >= 30 && core.len() <= 500);

    // Too short: same shape, under 30 characters.
    let short = "If you have. final answer.";
    assert!(short.len() < 30, "fixture must be under the floor");
    assert!(
        !is_client_meta_wrapper(short),
        "under 30 characters is not a wrapper"
    );

    // Exactly at the bounds.
    let at_floor = format!("{core}{}", " ".repeat(30usize.saturating_sub(core.len())));
    assert!(is_client_meta_wrapper(&at_floor));

    let padded = format!("{core}{}", "x".repeat(500 - core.len()));
    assert_eq!(padded.len(), 500);
    assert!(is_client_meta_wrapper(&padded), "exactly 500 is inside");

    let over = format!("{padded}x");
    assert_eq!(over.len(), 501);
    assert!(
        !is_client_meta_wrapper(&over),
        "past 500 characters this is a real message that quotes an instruction, \
         not an injected wrapper"
    );
}

/// Matching is case-insensitive and leading/trailing whitespace is trimmed
/// before the length check — a client that pretty-prints its wrapper must not
/// slip through.
#[test]
fn matching_ignores_case_and_surrounding_whitespace() {
    let variants = [
        "IF YOU HAVE COMPLETED THE WORK, CALL TASK_COMPLETE NOW.",
        "\n\n  If You Have completed the work, call Task_Complete now.  \n",
        "\t if the task is done and fully completed, stop here now.\t",
    ];
    for v in variants {
        assert!(is_client_meta_wrapper(v), "should match: {v:?}");
    }

    // Trimming applies to the length check too: padding a short string with
    // whitespace must not push it over the floor.
    let padded_short = format!(
        "{}{}{}",
        " ".repeat(50),
        "If you have. final answer.",
        " ".repeat(50)
    );
    assert!(
        !is_client_meta_wrapper(&padded_short),
        "whitespace must not count toward the 30-character floor"
    );
}

/// Ordinary prose that shares vocabulary with the heuristic must survive. This
/// is the table that fails if someone widens the predicate.
#[test]
fn ordinary_prose_is_not_deleted() {
    for text in [
        "Can you give me the final answer to the question about the tax rules?",
        "The task is complete, thanks for your help with the migration today.",
        "Once you have a moment I would like to discuss the deployment plan.",
        "If the task list is too long, feel free to summarise it for me instead.",
        "Please call the customer back when the report is fully completed.",
        "",
        "hi",
    ] {
        assert!(
            !is_client_meta_wrapper(text),
            "ordinary prose must survive: {text:?}"
        );
    }
}

/// The indexed strip is the variant callers use when they carry a per-turn
/// side table (image attachments). The returned indices must describe the
/// *input* positions of the survivors, or entry `i` ends up attached to a
/// different turn than it was before the strip.
#[test]
fn the_indexed_strip_returns_input_positions_of_survivors() {
    let wrapper_text = wrapper("If you have", "call task_complete now.");
    let mut turns = vec![
        ChatTurn::System("sys"),
        ChatTurn::User("real question one"),
        ChatTurn::User(&wrapper_text),
        ChatTurn::User("real question two"),
    ];
    let kept = strip_client_meta_wrappers_indexed(&mut turns);

    assert_eq!(turns.len(), 3, "exactly the wrapper should go");
    assert_eq!(
        kept,
        vec![0, 1, 3],
        "the surviving indices must be the INPUT positions, so a side table \
         filtered by them stays aligned"
    );
}

/// A wrapper in an assistant or system turn is not touched: the heuristic is
/// about client-injected *user* turns, and stripping a system prompt that
/// happens to match would remove the operator's instructions.
#[test]
fn only_user_turns_are_eligible() {
    let wrapper_text = wrapper("If you have", "call task_complete now.");
    let mut turns = vec![
        ChatTurn::System(&wrapper_text),
        ChatTurn::Assistant {
            text: &wrapper_text,
            tool_calls: &[],
        },
    ];
    let kept = strip_client_meta_wrappers_indexed(&mut turns);
    assert_eq!(turns.len(), 2, "non-user turns must survive verbatim");
    assert_eq!(kept, vec![0, 1]);
}

/// Nothing to strip is the common case and must return every index, not an
/// empty vector — a caller filtering its side table by an empty list would
/// drop every attachment.
#[test]
fn a_conversation_with_no_wrappers_keeps_every_index() {
    let mut turns = vec![
        ChatTurn::System("sys"),
        ChatTurn::User("hello"),
        ChatTurn::User("world"),
    ];
    let kept = strip_client_meta_wrappers_indexed(&mut turns);
    assert_eq!(turns.len(), 3);
    assert_eq!(kept, vec![0, 1, 2]);

    // And an empty conversation is an empty list, not a panic.
    let mut empty: Vec<ChatTurn<'_>> = Vec::new();
    assert!(strip_client_meta_wrappers_indexed(&mut empty).is_empty());
}
