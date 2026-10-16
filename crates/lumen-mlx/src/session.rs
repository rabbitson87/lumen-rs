//! Conversation sessions: reusing a conversation's KV from one turn to the next.
//!
//! Nothing here depends on the model. A family plugs in with its runner (an
//! implementation of [`SessionRunner`]) and the token ids of its generation
//! header (what [`conversation_boundary`] cuts at, encoded only when rollback
//! points are on). A turn is then the same few calls around the family's own
//! decode loop:
//!
//! ```ignore
//! sessions.evict_stale(runner);
//! let key = sessions.resolve_key(explicit_id, &prompt);
//! let boundary = session::conversation_boundary(&prompt, || encode(header).ok());
//! let start = sessions.position(runner, &key, &prompt, "", || alloc_seq())?;
//! let (first, pos, point) = session::feed(runner, &key, &start, &prompt, boundary, "")?;
//! // … decode from `first` at `pos` on `start.seq_id` …
//! sessions.store(runner, &key, start.seq_id, prompt_and_generated, point);
//! ```
//!
//! What a runner gets for what it implements:
//!
//! * `prefill`, `extend`, `remove_seq` — every runner has them. With only these,
//!   a turn that extends the previous one exactly (prompt + reply re-rendered
//!   byte for byte) reuses its KV.
//! * `prefill_grid`, `mark_rollback`, `rollback` — opt-in, refused by default,
//!   and implemented together: a grid is what makes [`feed`] place a point, so
//!   a grid without the rollback pair fails the turn at its first mark. With
//!   all three, a turn whose history drops the previous generation header —
//!   most chat templates do — resumes from the conversation boundary instead of
//!   prefilling everything again. For a `NativePromptCache` the rollback pair is
//!   `native_snapshot::RollbackPoint::capture` / `restore_into`, and
//!   `prefill_grid` says how the family's bulk prefill is chunked — points go
//!   on its chunk boundaries, the only exact cuts (see `session_feed`).

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::session_feed::{self, PrefillGrid};

/// What the session machinery needs from a family's runner.
///
/// A narrow view on purpose, like `prefix_cache::SnapshotRunner`: a family
/// implements these on whatever runner it has, and nothing else of it leaks in.
pub(crate) trait SessionRunner {
    /// Start `seq_id` with `tokens`; returns the next token and the position.
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    /// Append `tokens` to `seq_id`; returns the next token and the position.
    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    /// Drop `seq_id` and its KV.
    fn remove_seq(&mut self, seq_id: u64) -> Result<()>;
    /// The shape of a bulk prefill of `n` tokens. `None` — the default —
    /// means the runner cannot say, and then no prefill is ever cut to place
    /// a rollback point, because nothing could tell whether the cut is exact.
    /// A runner that returns a grid must implement the rollback pair too.
    fn prefill_grid(&self, _n: usize) -> Option<PrefillGrid> {
        None
    }
    /// Remember `seq_id`'s current state as the point [`Self::rollback`]
    /// returns to, replacing any earlier one.
    fn mark_rollback(&mut self, _seq_id: u64) -> Result<()> {
        Err(anyhow!("this runner has no rollback points"))
    }
    /// Put `seq_id` back to its rollback point and return that position. The
    /// point stays, so the same position can be returned to again.
    fn rollback(&mut self, _seq_id: u64) -> Result<usize> {
        Err(anyhow!("this runner has no rollback points"))
    }
}

/// Per-session prompt-cache state. Holds the running token sequence (prompt +
/// generated) so that the next request on the same session can skip prefilling
/// the longest common prefix. `last_access` is set each time the turn is
/// stored, so TTL + LRU eviction can run from a single timestamp.
pub(crate) struct SessionState {
    pub(crate) seq_id: u64,
    pub(crate) tokens: Vec<u32>,
    /// How many of `tokens` the runner can wind this sequence back to: the
    /// rollback point left at the conversation boundary, before the generation
    /// header. `None` when no point was placed.
    pub(crate) rollback_len: Option<usize>,
    pub(crate) last_access: Instant,
}

impl SessionState {
    /// Is this session exactly `prompt_ids`, one turn earlier?
    ///
    /// `tokens` is what the model was fed plus what it generated, so a strict
    /// prefix match means the KV behind this sequence is a prefix of the KV this
    /// prompt needs — the suffix can be appended instead of re-prefilling the
    /// whole conversation. It is also the *only* thing that makes reuse safe, so
    /// key selection can be a guess: a wrong guess costs a prefill, never an
    /// answer.
    pub(crate) fn extends(&self, prompt_ids: &[u32]) -> bool {
        !self.tokens.is_empty()
            && prompt_ids.len() > self.tokens.len()
            && prompt_ids.starts_with(&self.tokens)
    }

    /// Where this prompt can pick the session up from its rollback point,
    /// when it does not extend the whole session.
    ///
    /// That is the usual case for a chat turn, not an edge: the generation
    /// header a turn ends with never comes back in the next prompt. Qwen 3.5
    /// and 3.6 templates drop the `<think>` block from replayed assistant
    /// turns, and on any checkpoint a client that does not return the trace
    /// drops it too. So `extends` misses on every turn of a plain chat, while
    /// everything before the header — the point kept here — is reproduced.
    /// Same guard as `extends`: the prompt must actually continue those
    /// tokens, so a wrong guess costs a prefill, never an answer.
    pub(crate) fn resumes_at(&self, prompt_ids: &[u32]) -> Option<usize> {
        let kept = self.rollback_len?;
        (kept > 0
            && kept <= self.tokens.len()
            && prompt_ids.len() > kept
            && prompt_ids.starts_with(&self.tokens[..kept]))
        .then_some(kept)
    }

    /// How many of this prompt's tokens the session already holds, by
    /// whichever route reaches more of them.
    pub(crate) fn reusable_len(&self, prompt_ids: &[u32]) -> Option<usize> {
        if self.extends(prompt_ids) {
            Some(self.tokens.len())
        } else {
            self.resumes_at(prompt_ids)
        }
    }
}

/// How a session turn's sequence got to where its prompt starts feeding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionReuse {
    /// The prompt continues the whole session; its end is the start.
    Extend,
    /// The prompt continues the session only up to its rollback point, and
    /// the sequence was wound back there.
    Resume,
    /// Nothing reusable; a new, empty sequence.
    Fresh,
}

/// A session turn's sequence, positioned for its prompt to be fed.
pub(crate) struct SessionStart {
    pub(crate) seq_id: u64,
    /// Prompt tokens the sequence already holds.
    pub(crate) from: usize,
    pub(crate) how: SessionReuse,
    /// The rollback point the sequence still holds, in tokens.
    pub(crate) point: Option<usize>,
}

/// Key prefix for sessions the server invented rather than the client naming.
const AUTO_SESSION_PREFIX: &str = "auto/";

/// How many server-invented sessions to keep alive.
///
/// Bounded where an explicit session is not, and deliberately: a client that
/// names a session has said it will come back, while an auto session is our
/// guess, and each one holds a whole conversation's KV until it is evicted. Two
/// covers the case this exists for — a conversation growing turn by turn — plus
/// one alternate interleaved with it. `LUMEN_MLX_SESSION_MAX` still caps the
/// pool as a whole for an operator who wants a harder bound.
const AUTO_SESSION_MAX: usize = 2;

lumen_flags::flag! {
    /// Reuse a conversation's KV without the client naming a session.
    ///
    /// Neither the OpenAI nor the Anthropic request has a conversation id, and
    /// `session_id` is a Lumen extension no stock client sends — so without
    /// this, every spec-conformant client re-prefilled its whole conversation
    /// on every turn. ON matches the prompt against live sessions by token
    /// prefix instead, which needs no cooperation and cannot be wrong: the same
    /// `starts_with` gate guards the reuse itself.
    ///
    /// **`Behavior`, not `Optimization`, and the mechanism is measured rather
    /// than assumed.** `session_reuse_reproduces_a_cold_prefill` runs three
    /// arms over the same tokens and localizes it: `extend` reproduces a bulk
    /// prefill **bit-identically**, so the cache handoff is exact. What is not
    /// exact is the decode path in between — the tokens between two prompts
    /// advanced the cache one at a time, and a single-token forward is a
    /// different matmul shape (M=1, GEMV) from a bulk chunk (M=N, GEMM), so the
    /// reduction order differs. Measured on Qwen3.8-27B: identical for 43
    /// tokens, then a near-tie argmax flips at 44. (Chunk invariance does not
    /// cover this — 256 vs 2048 are both GEMM.)
    ///
    /// Each path is deterministic and stable across runs; they just are not
    /// each other, so the equivalence matrix must never flip this. Not a new
    /// property — the explicit-`session_id` path has always had it, and this
    /// output is byte-identical to that path's.
    pub(crate) auto_session_enabled {
        env: "LUMEN_MLX_AUTO_SESSION",
        default: true,
        kind: Behavior,
    }
}

lumen_flags::flag! {
    /// Leave a rollback point at each chat turn's conversation boundary, so
    /// the next turn can resume a session it does not exactly extend.
    ///
    /// Without it, plain multi-turn chat on Qwen 3.5 / 3.6 never reuses a
    /// session: the generation header a turn ends with is not in the next
    /// prompt (the template drops the `<think>` block from replayed turns), so
    /// every turn prefilled the whole conversation again — measured on
    /// Qwen3.5-9B with an 11.5K-token system prompt, about 25 s a turn. The
    /// same holds on 3.8 for any client that does not return the trace.
    ///
    /// **`Optimization`: the output does not change, by construction.** The
    /// point is placed by cutting the prefill, and `session_feed::plan_feed`
    /// only cuts on the bulk pass's own chunk grid, so every call is one of
    /// that pass's chunks. A resumed turn re-feeds what
    /// follows the point through the same bulk path, which is what a cold
    /// prefill of the longer prompt does — `extend` reproduces a bulk prefill
    /// bit-identically (`session_reuse_reproduces_a_cold_prefill`).
    pub(crate) session_rollback_enabled {
        env: "LUMEN_MLX_SESSION_ROLLBACK",
        default: true,
        kind: Optimization,
    }
}

/// Read session-eviction limits from env. Both default to "no limit".
///
/// - `LUMEN_MLX_SESSION_TTL_SECS=N` — drop sessions idle > N seconds.
/// - `LUMEN_MLX_SESSION_MAX=N`      — keep at most N sessions, evicting
///   least-recently-used first.
fn read_session_limits() -> (Option<Duration>, Option<usize>) {
    let ttl = std::env::var("LUMEN_MLX_SESSION_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(Duration::from_secs);
    let max = std::env::var("LUMEN_MLX_SESSION_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    (ttl, max)
}

/// Where this chat prompt's conversation ends and its generation header
/// begins — the part the next turn's history renders again.
///
/// `header` returns the family's generation header, tokenized on its own
/// (Qwen: `<|im_start|>assistant\n<think>…`); it is only called when rollback
/// points are on. `None` when they are off, the header cannot be encoded, or
/// it is not a clean token suffix of the prompt (then there is no boundary to
/// trust, and the turn simply gets no rollback point).
pub(crate) fn conversation_boundary(
    prompt_ids: &[u32],
    header: impl FnOnce() -> Option<Vec<u32>>,
) -> Option<usize> {
    if !session_rollback_enabled::get() {
        return None;
    }
    let header = header()?;
    (!header.is_empty() && prompt_ids.len() > header.len() && prompt_ids.ends_with(&header))
        .then(|| prompt_ids.len() - header.len())
}

/// Feed `prompt[start.from..]` to the positioned sequence — a prefill from
/// position 0, an extend otherwise — leaving a rollback point at `boundary`
/// when [`session_feed::plan_feed`] finds an exact place for one. Returns the
/// next token, the position, and the rollback point the sequence now holds
/// (the new one, or the one it already had).
pub(crate) fn feed<R: SessionRunner + ?Sized>(
    runner: &mut R,
    key: &str,
    start: &SessionStart,
    prompt: &[u32],
    boundary: Option<usize>,
    label: &str,
) -> Result<(u32, usize, Option<usize>)> {
    let (seq_id, from, end) = (start.seq_id, start.from, prompt.len());
    let plan = match (boundary, runner.prefill_grid(end)) {
        (Some(b), Some(grid)) => session_feed::plan_feed(from, end, grid, b),
        _ => session_feed::FeedPlan {
            cuts: vec![end],
            mark: None,
        },
    };
    let t = Instant::now();
    let (mut last, mut pos, mut at) = (0, 0, from);
    for &to in &plan.cuts {
        (last, pos) = if at == 0 {
            runner.prefill(seq_id, &prompt[at..to])?
        } else {
            runner.extend(seq_id, &prompt[at..to])?
        };
        if plan.mark == Some(to) {
            runner.mark_rollback(seq_id)?;
        }
        at = to;
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let marked = plan
        .mark
        .map(|m| format!(", rollback point at {m}"))
        .unwrap_or_default();
    let fed = end - from;
    match start.how {
        SessionReuse::Extend => eprintln!(
            "[mlx] seq {seq_id} session={key:?}{label} reuse: cached={from} \
             suffix={fed} in {ms:.0}ms{marked}"
        ),
        SessionReuse::Resume => eprintln!(
            "[mlx] seq {seq_id} session={key:?}{label} resume: rolled back to {from} \
             suffix={fed} in {ms:.0}ms{marked}"
        ),
        SessionReuse::Fresh => eprintln!(
            "[mlx] seq {seq_id} session={key:?}{label} fresh prefill: {fed} tokens \
             in {ms:.0}ms{marked}"
        ),
    }
    Ok((last, pos, plan.mark.or(start.point)))
}

/// The live sessions of one backend, and the policy for keying and evicting
/// them.
pub(crate) struct Sessions {
    map: HashMap<String, SessionState>,
    /// Monotonic counter for `auto/N` session keys.
    auto_seq: u64,
    /// Cap on server-invented sessions; `0` disables them.
    auto_max: usize,
    ttl: Option<Duration>,
    max: Option<usize>,
}

impl Sessions {
    /// No TTL or LRU cap; auto sessions as `LUMEN_MLX_AUTO_SESSION` says.
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_limits(None, None)
    }

    /// Limits from `LUMEN_MLX_SESSION_TTL_SECS` / `LUMEN_MLX_SESSION_MAX`,
    /// logged when set.
    pub(crate) fn from_env() -> Self {
        let (ttl, max) = read_session_limits();
        if let Some(ttl) = ttl {
            eprintln!("[mlx] session TTL = {}s", ttl.as_secs());
        }
        if let Some(max) = max {
            eprintln!("[mlx] session LRU cap = {max}");
        }
        Self::with_limits(ttl, max)
    }

    fn with_limits(ttl: Option<Duration>, max: Option<usize>) -> Self {
        Self {
            map: HashMap::new(),
            auto_seq: 0,
            auto_max: if auto_session_enabled::get() {
                AUTO_SESSION_MAX
            } else {
                0
            },
            ttl,
            max,
        }
    }

    /// Number of live sessions.
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    pub(crate) fn get(&self, key: &str) -> Option<&SessionState> {
        self.map.get(key)
    }

    /// Which session key this request should use.
    ///
    /// An explicit id is taken as given — the client is managing the session and
    /// gets exactly the behaviour it always had. Without one, the prompt itself
    /// identifies the conversation: see [`longest_prefix_session`]. Only when
    /// nothing matches is a new `auto/N` key minted, and only if auto sessions
    /// are enabled at all.
    ///
    /// Returning a fresh key when auto sessions are off is what keeps the old
    /// behaviour intact: nothing will ever match it, so the turn cold-prefills
    /// and the auto-session cap drops it again immediately.
    pub(crate) fn resolve_key(&mut self, explicit: Option<&str>, prompt_ids: &[u32]) -> String {
        if let Some(id) = explicit {
            return id.to_string();
        }
        if self.auto_max > 0
            && let Some(k) = longest_prefix_session(&self.map, prompt_ids)
        {
            return k;
        }
        self.auto_seq += 1;
        format!("{AUTO_SESSION_PREFIX}{}", self.auto_seq)
    }

    /// Put the sequence for session `key` where `prompt_ids` can continue it.
    ///
    /// The session's end when the prompt extends all of it; its rollback point
    /// when the prompt continues only that far, winding the sequence back
    /// there; otherwise a new, empty sequence from `alloc_seq`, dropping the old
    /// one. A rollback that fails is logged and treated as no match — the turn
    /// still gets a correct answer, only without the reuse.
    pub(crate) fn position<R: SessionRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        key: &str,
        prompt_ids: &[u32],
        label: &str,
        alloc_seq: impl FnOnce() -> u64,
    ) -> Result<SessionStart> {
        let found = self.map.get(key).map(|s| {
            (
                s.seq_id,
                s.extends(prompt_ids).then_some(s.tokens.len()),
                s.resumes_at(prompt_ids),
                s.rollback_len,
            )
        });
        match found {
            Some((seq_id, Some(len), _, point)) => {
                return Ok(SessionStart {
                    seq_id,
                    from: len,
                    how: SessionReuse::Extend,
                    point,
                });
            }
            Some((seq_id, None, Some(kept), _)) => match runner.rollback(seq_id) {
                Ok(pos) if pos == kept => {
                    return Ok(SessionStart {
                        seq_id,
                        from: kept,
                        how: SessionReuse::Resume,
                        point: Some(kept),
                    });
                }
                other => eprintln!(
                    "[mlx] session={key:?}{label} rollback to {kept} failed ({:?}); prefilling fresh",
                    other.map_err(|e| format!("{e:#}"))
                ),
            },
            _ => {}
        }
        // No usable session → drop any old one, alloc new.
        if let Some(old) = self.map.remove(key) {
            let _ = runner.remove_seq(old.seq_id);
            eprintln!(
                "[mlx] session={key:?}{label} divergent prompt; dropping old seq {} ({} tokens)",
                old.seq_id,
                old.tokens.len()
            );
        }
        Ok(SessionStart {
            seq_id: alloc_seq(),
            from: 0,
            how: SessionReuse::Fresh,
            point: None,
        })
    }

    /// Persist the turn: `tokens` (prompt + everything generated) behind
    /// `seq_id`, with its rollback point. The sequence is kept, not removed —
    /// its KV is what the next turn reuses.
    pub(crate) fn store<R: SessionRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        key: &str,
        seq_id: u64,
        tokens: Vec<u32>,
        rollback_len: Option<usize>,
    ) {
        self.map.insert(
            key.to_string(),
            SessionState {
                seq_id,
                tokens,
                rollback_len,
                last_access: Instant::now(),
            },
        );
        self.evict_excess_auto(runner);
    }

    /// Drop a session's prompt cache. Returns true if the session existed.
    pub(crate) fn remove<R: SessionRunner + ?Sized>(&mut self, runner: &mut R, key: &str) -> bool {
        if let Some(state) = self.map.remove(key) {
            let _ = runner.remove_seq(state.seq_id);
            true
        } else {
            false
        }
    }

    /// Drop sessions idle past TTL, then enforce LRU cap if set. No-op when
    /// neither limit is configured. Called at the entry of every session-aware
    /// generation request.
    pub(crate) fn evict_stale<R: SessionRunner + ?Sized>(&mut self, runner: &mut R) {
        let now = Instant::now();
        for (key, reason) in pick_eviction_victims(&self.map, now, self.ttl, self.max) {
            if let Some(state) = self.map.remove(&key) {
                let _ = runner.remove_seq(state.seq_id);
                eprintln!("[mlx] session={key:?} evicted ({reason})");
            }
        }
    }

    /// Drop auto sessions past the cap, oldest first, releasing their KV.
    fn evict_excess_auto<R: SessionRunner + ?Sized>(&mut self, runner: &mut R) {
        for key in pick_auto_session_victims(&self.map, self.auto_max) {
            if let Some(state) = self.map.remove(&key) {
                let _ = runner.remove_seq(state.seq_id);
                eprintln!(
                    "[mlx] session={key:?} evicted (auto-session cap {}); freed seq {} ({} tokens)",
                    self.auto_max,
                    state.seq_id,
                    state.tokens.len()
                );
            }
        }
    }
}

/// Which live session a prompt should extend, when the client named none.
///
/// A session's `tokens` are what the model was fed plus what it generated, so a
/// session whose tokens are a strict prefix of this prompt **is** this
/// conversation one turn earlier — no id required, and none available: neither
/// the OpenAI nor the Anthropic request has a conversation identifier, and
/// `session_id` is a Lumen extension no stock client sends. Matching on the
/// tokens instead is what makes reuse reachable from a spec-conformant request.
///
/// The longest match wins, so a conversation that has grown extends its own
/// newest state rather than an older snapshot of itself. A session also
/// matches up to its rollback point (see [`SessionState::resumes_at`]), which
/// is how a plain chat turn finds the previous one at all.
///
/// Explicit sessions are candidates too: a client that named a session on turn
/// one and omitted it on turn two still continues the same conversation.
fn longest_prefix_session(
    sessions: &HashMap<String, SessionState>,
    prompt_ids: &[u32],
) -> Option<String> {
    sessions
        .iter()
        .filter_map(|(k, s)| s.reusable_len(prompt_ids).map(|n| (k, n)))
        .max_by_key(|&(_, n)| n)
        .map(|(k, _)| k.clone())
}

/// LRU victims among the **auto** sessions only, once they exceed `max`.
///
/// Explicit sessions are exempt: naming one is the client saying it will come
/// back, while an auto session is our guess, and each holds a whole
/// conversation's KV until dropped. `max == 0` means auto sessions are off, in
/// which case every one of them is a victim.
fn pick_auto_session_victims(sessions: &HashMap<String, SessionState>, max: usize) -> Vec<String> {
    let mut autos: Vec<(&String, Instant)> = sessions
        .iter()
        .filter(|(k, _)| k.starts_with(AUTO_SESSION_PREFIX))
        .map(|(k, s)| (k, s.last_access))
        .collect();
    if autos.len() <= max {
        return Vec::new();
    }
    // Oldest first, drop until the cap is met.
    autos.sort_by_key(|&(_, t)| t);
    autos[..autos.len() - max]
        .iter()
        .map(|(k, _)| (*k).clone())
        .collect()
}

/// Pure helper for [`Sessions::evict_stale`]: picks (key, human_reason) pairs
/// for every session that should be dropped at this moment. TTL victims first,
/// then LRU until `session_max` is satisfied. Pulled out so the policy is unit-
/// testable without spinning up a real runner.
fn pick_eviction_victims(
    sessions: &HashMap<String, SessionState>,
    now: Instant,
    session_ttl: Option<Duration>,
    session_max: Option<usize>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut alive: HashMap<String, Instant> = sessions
        .iter()
        .map(|(k, s)| (k.clone(), s.last_access))
        .collect();

    if let Some(ttl) = session_ttl {
        let stale: Vec<String> = alive
            .iter()
            .filter(|&(_, t)| now.duration_since(*t) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale {
            alive.remove(&k);
            out.push((k, format!("TTL > {}s", ttl.as_secs())));
        }
    }

    if let Some(max) = session_max {
        while alive.len() > max {
            let victim = alive
                .iter()
                .min_by_key(|&(_, t)| *t)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    alive.remove(&k);
                    out.push((k, format!("LRU cap {max}")));
                }
                None => break,
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_session(seq_id: u64, last_access: Instant) -> SessionState {
        SessionState {
            seq_id,
            tokens: vec![1, 2, 3],
            rollback_len: None,
            last_access,
        }
    }

    fn session_with(tokens: Vec<u32>, last_access: Instant) -> SessionState {
        SessionState {
            seq_id: 0,
            tokens,
            rollback_len: None,
            last_access,
        }
    }

    /// A stock OpenAI or Anthropic request carries no conversation id, and
    /// `session_id` is a Lumen extension no such client sends — so before this,
    /// every one of them re-prefilled the whole conversation, every turn.
    ///
    /// The prompt identifies its own conversation instead: a session's tokens
    /// are what the model was fed plus what it generated, so a strict prefix
    /// match is that conversation, one turn earlier.
    #[test]
    fn a_prompt_finds_its_own_conversation_without_being_told_which() {
        let now = Instant::now();
        let mut map = HashMap::new();
        map.insert("auto/1".to_string(), session_with(vec![1, 2, 3], now));
        // A longer state of the SAME conversation must win, so a growing chat
        // extends its newest turn rather than an older snapshot of itself.
        map.insert("auto/2".to_string(), session_with(vec![1, 2, 3, 4, 5], now));
        // …and an unrelated conversation must not match at all.
        map.insert("auto/3".to_string(), session_with(vec![9, 9], now));

        assert_eq!(
            longest_prefix_session(&map, &[1, 2, 3, 4, 5, 6]).as_deref(),
            Some("auto/2")
        );
        // An explicit session is a candidate too — a client may name one on
        // turn 1 and omit it afterwards.
        map.insert(
            "mine".to_string(),
            session_with(vec![1, 2, 3, 4, 5, 6], now),
        );
        assert_eq!(
            longest_prefix_session(&map, &[1, 2, 3, 4, 5, 6, 7]).as_deref(),
            Some("mine")
        );
    }

    /// The guard that makes guessing a key safe at all.
    #[test]
    fn a_session_only_extends_a_prompt_that_actually_continues_it() {
        let now = Instant::now();
        let s = session_with(vec![1, 2, 3], now);
        assert!(s.extends(&[1, 2, 3, 4]));
        // Identical, not longer: there is nothing to append, and re-running it
        // as an extension would decode from the wrong position.
        assert!(!s.extends(&[1, 2, 3]));
        assert!(!s.extends(&[1, 2]));
        // Divergent at the last token — the case a system-hash key would get
        // wrong and this cannot.
        assert!(!s.extends(&[1, 2, 4, 5]));
        // An empty session has no KV to reuse.
        assert!(!session_with(Vec::new(), now).extends(&[1, 2, 3]));

        let mut map = HashMap::new();
        map.insert("auto/1".to_string(), s);
        assert!(longest_prefix_session(&map, &[7, 8, 9]).is_none());
    }

    /// A chat session matches the next turn up to its rollback point even when
    /// the turn does not extend it — and only if the turn really continues
    /// those tokens.
    #[test]
    fn a_session_is_found_again_through_its_rollback_point() {
        let now = Instant::now();
        // Turn one: 4 conversation tokens, a 2-token generation header (8, 9),
        // a 1-token reply. The point sits before the header.
        let mut s = session_with(vec![1, 2, 3, 4, 8, 9, 50], now);
        s.rollback_len = Some(4);

        // Turn two re-renders the reply without the header: no extension…
        let next = [1, 2, 3, 4, 50, 60, 8, 9];
        assert!(!s.extends(&next));
        // …but everything up to the point is there.
        assert_eq!(s.resumes_at(&next), Some(4));
        assert_eq!(s.reusable_len(&next), Some(4));
        // A prompt that diverges before the point does not match.
        assert_eq!(s.resumes_at(&[1, 2, 7, 4, 50]), None);
        // Nothing past the point: nothing to feed.
        assert_eq!(s.resumes_at(&[1, 2, 3, 4]), None);
        // Exact extension still wins when the prompt has it.
        let extended = [1, 2, 3, 4, 8, 9, 50, 70];
        assert_eq!(s.reusable_len(&extended), Some(7));
        // A session without a point has only the exact route.
        assert_eq!(
            session_with(vec![1, 2, 3, 4, 8, 9], now).resumes_at(&next),
            None
        );

        let mut map = HashMap::new();
        map.insert("auto/1".to_string(), s);
        map.insert("auto/2".to_string(), session_with(vec![1, 2], now));
        assert_eq!(
            longest_prefix_session(&map, &next).as_deref(),
            Some("auto/1"),
            "the resumable session reuses more than the shorter exact one"
        );
    }

    /// The boundary is wherever the family's header sits at the end of the
    /// prompt — and nowhere if it does not, so a header the template renders
    /// differently costs the rollback point, never a wrong cut.
    #[test]
    fn the_boundary_is_where_the_family_s_header_starts() {
        let header = || Some(vec![8, 9]);
        session_rollback_enabled::with(true, || {
            assert_eq!(conversation_boundary(&[1, 2, 3, 8, 9], header), Some(3));
            assert_eq!(conversation_boundary(&[1, 2, 3, 8], header), None);
            assert_eq!(conversation_boundary(&[8, 9], header), None);
            assert_eq!(conversation_boundary(&[1, 8, 9], || Some(Vec::new())), None);
            assert_eq!(conversation_boundary(&[1, 8, 9], || None), None);
        });
        session_rollback_enabled::with(false, || {
            // Off means no point and no encode: the header is not even asked for.
            let unasked = || -> Option<Vec<u32>> { panic!("header encoded with rollback off") };
            assert_eq!(conversation_boundary(&[1, 2, 3, 8, 9], unasked), None);
        });
    }

    /// Auto sessions are capped where explicit ones are not: naming a session
    /// is the client saying it will come back, while an auto one is our guess,
    /// and each holds a whole conversation's KV until dropped.
    #[test]
    fn only_the_sessions_the_server_invented_are_capped() {
        let now = Instant::now();
        let mut map = HashMap::new();
        map.insert(
            "auto/1".to_string(),
            session_with(vec![1], now - Duration::from_secs(30)),
        );
        map.insert(
            "auto/2".to_string(),
            session_with(vec![2], now - Duration::from_secs(20)),
        );
        map.insert(
            "auto/3".to_string(),
            session_with(vec![3], now - Duration::from_secs(10)),
        );
        map.insert(
            "named".to_string(),
            session_with(vec![4], now - Duration::from_secs(99)),
        );

        // Oldest auto first; the named one is exempt however stale it is.
        assert_eq!(
            pick_auto_session_victims(&map, 2),
            vec!["auto/1".to_string()]
        );
        assert_eq!(
            pick_auto_session_victims(&map, 1),
            vec!["auto/1".to_string(), "auto/2".to_string()]
        );
        // `0` disables auto sessions — every one of them goes, the named one stays.
        assert_eq!(
            pick_auto_session_victims(&map, 0),
            vec![
                "auto/1".to_string(),
                "auto/2".to_string(),
                "auto/3".to_string()
            ]
        );
        assert!(pick_auto_session_victims(&map, 8).is_empty());
    }

    #[test]
    fn pick_eviction_victims_no_limits_keeps_all() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert("a".to_string(), fake_session(1, now));
        map.insert("b".to_string(), fake_session(2, now));
        let victims = pick_eviction_victims(&map, now, None, None);
        assert!(victims.is_empty());
    }

    #[test]
    fn pick_eviction_victims_ttl_drops_stale_only() {
        let mut map = HashMap::new();
        let now = Instant::now();
        let stale = now - Duration::from_secs(60);
        map.insert("fresh".to_string(), fake_session(1, now));
        map.insert("stale".to_string(), fake_session(2, stale));

        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(10)), None);
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["stale"]);
        assert!(victims[0].1.contains("TTL"));
    }

    #[test]
    fn pick_eviction_victims_lru_drops_oldest_until_cap() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert(
            "oldest".to_string(),
            fake_session(1, now - Duration::from_secs(30)),
        );
        map.insert(
            "middle".to_string(),
            fake_session(2, now - Duration::from_secs(10)),
        );
        map.insert("newest".to_string(), fake_session(3, now));

        let victims = pick_eviction_victims(&map, now, None, Some(1));
        // Expect 2 victims (3 - 1 cap), oldest first.
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0].0, "oldest");
        assert_eq!(victims[1].0, "middle");
        assert!(victims[0].1.contains("LRU"));
    }

    #[test]
    fn pick_eviction_victims_ttl_then_lru() {
        // TTL drops one, LRU drops another from the survivors.
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert(
            "expired".to_string(),
            fake_session(1, now - Duration::from_secs(120)),
        );
        map.insert(
            "oldish".to_string(),
            fake_session(2, now - Duration::from_secs(20)),
        );
        map.insert("fresh".to_string(), fake_session(3, now));

        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(60)), Some(1));
        // expired: TTL. oldish: LRU (fresh survives since cap=1).
        assert_eq!(victims.len(), 2);
        let reasons: Vec<&str> = victims.iter().map(|(_, r)| r.as_str()).collect();
        assert!(reasons.iter().any(|r| r.contains("TTL")));
        assert!(reasons.iter().any(|r| r.contains("LRU")));
    }
}
