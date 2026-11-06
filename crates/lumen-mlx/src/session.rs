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
//! let start = sessions.position(runner, &key, &prompt, &turns, "", || alloc_seq())?;
//! let (first, pos, point) = session::feed(runner, &key, &start, &prompt, boundary, &turns, "")?;
//! // … decode from `first` at `pos` on `start.seq_id` …
//! sessions.store(runner, &key, start.seq_id, session::held_tokens(prompt, &generated, pos), point);
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
    /// Where a bulk prefill cuts its chunks. `None` — the default — means the
    /// runner cannot say, and then no prefill is ever cut to place a rollback
    /// point, because nothing could tell whether the cut is exact. A runner
    /// that returns a grid must implement the rollback pair too.
    fn prefill_grid(&self) -> Option<PrefillGrid> {
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
    /// Feed `tokens` (the start of `seq_id` when `fresh`, else its continuation) in one call,
    /// cutting the prefill on the runner's grid and leaving the rollback point `mark` tokens in:
    /// what feeding grid chunk by grid chunk and calling [`Self::mark_rollback`] at `mark` would
    /// do, for a runner that does better with the whole run at once (it reads each layer's
    /// weights once for all the chunks). `None` — the default — means it does not, and the caller
    /// feeds cut by cut.
    fn feed_marked(
        &mut self,
        _seq_id: u64,
        _fresh: bool,
        _tokens: &[u32],
        _mark: usize,
    ) -> Result<Option<(u32, usize)>> {
        Ok(None)
    }
    /// Write `seq_id`'s rollback point — the state after `prefix`, a cut of
    /// the grid `grid_key` names ([`PrefillGrid::key_at`]) — to the runner's
    /// disk tier. Returns whether anything was written. Default: the runner
    /// has no disk tier.
    fn persist_point(&mut self, _seq_id: u64, _prefix: &[u32], _grid_key: u64) -> Result<bool> {
        Ok(false)
    }
    /// Start the new `seq_id` from the longest persisted state `prompt`
    /// continues, with a rollback point there; returns its length. `turns` is
    /// the prompt's turn cuts ([`session_feed::plan_turns`]); empty, the states
    /// are on the grid. Default: the runner has no disk tier, so never.
    fn restore_point(
        &mut self,
        _seq_id: u64,
        _prompt: &[u32],
        _turns: &[usize],
    ) -> Result<Option<usize>> {
        Ok(None)
    }
    /// Bytes by which a feed that leaves `kv_tokens` tokens of KV (counting every copy that
    /// will exist: a sequence's own and its rollback point's are two) and `states` per-sequence
    /// states in memory would take the runner past the memory it is planned to stay within,
    /// counting everything it holds now. `0` — the default — means it fits, or that the runner
    /// keeps no such budget. [`Sessions`] answers a shortfall by dropping the sessions it can
    /// spare before the feed starts.
    fn memory_shortfall(&self, _kv_tokens: usize, _states: usize) -> u64 {
        0
    }
}

/// What feeding the last `prompt_len - from` tokens of a prompt to a session's sequence leaves
/// in memory, for [`SessionRunner::memory_shortfall`]: the sequence and its rollback point both
/// hold the new KV, and a new sequence (`from == 0`) holds two states.
pub(crate) fn feed_cost(from: usize, prompt_len: usize) -> (usize, usize) {
    if from == 0 {
        (2 * prompt_len, 2)
    } else {
        (2 * prompt_len.saturating_sub(from), 0)
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
    /// The grid the rollback point is a cut of. A resume from it is exact
    /// only on that same grid, so a runner now on another grid prefills cold.
    pub(crate) point_grid: Option<PrefillGrid>,
    pub(crate) last_access: Instant,
}

impl SessionState {
    /// Is this session exactly `prompt_ids`, one turn earlier?
    ///
    /// `tokens` is what the sequence was fed: the prompt and the generated tokens
    /// it holds ([`held_tokens`]), so a strict prefix match means the KV
    /// behind this sequence is a prefix of the KV this prompt needs — the suffix can be appended instead of re-prefilling the
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
    /// No live session, but a state persisted by an earlier turn (or an
    /// earlier process) that the prompt continues; a new sequence restored
    /// from it.
    Restored,
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
    /// **`Behavior`, not `Optimization`.** `session_reuse_reproduces_a_cold_prefill`
    /// runs three arms over the same tokens: `extend` reproduces a bulk prefill
    /// **bit-identically**, so the cache handoff is exact. What a session adds
    /// is the decode path in between: the tokens between two prompts advanced
    /// the cache one at a time (M=1) rather than in a bulk chunk (M=N), a
    /// different matmul, so their keys and values need not equal a prefill's in
    /// every bit, and a near-tie could flip. Measured with the arms fed the same
    /// tokens: all 48 tokens equal a cold prefill's on Qwen3.6-35B-A3B,
    /// Qwen3.6-27B and Qwen3.8-27B. (An earlier "flips at 44" was the test
    /// leaving out the reply's last token, which the session store also did:
    /// see `held_tokens`.)
    ///
    /// Each path is deterministic and stable across runs, but they are not
    /// guaranteed to be each other, so the equivalence matrix must never flip
    /// this. Not a new property — the explicit-`session_id` path has always had
    /// it, and this output is byte-identical to that path's.
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

lumen_flags::flag! {
    /// Drop the conversations a memory-planned runner cannot hold beside a new prompt.
    ///
    /// A runner that plans its memory (Flash-Next, whose expert pool takes what the GPU's memory
    /// limit leaves after one prompt's worth of KV) has no room for a second long conversation
    /// next to the one it is feeding. A session keeps its KV and a rollback point (two copies of
    /// the prompt's KV, ~2 GB at 16K tokens), and the tools path keeps a snapshot or two of every
    /// system prompt it has seen (~0.8 GB each at 12K tokens, without a bound). Past the memory
    /// limit MLX stops recycling its buffer cache, every allocation of every decode step becomes
    /// a fresh Metal buffer, and decode ran at half speed (3.8 against 7.7 tok/s, measured) for
    /// the second long prompt and every one after it; the prefill ran 15-35% slower too.
    ///
    /// ON, before a prompt is fed and only if it would not fit, drops the least recently used
    /// other sessions; then the prefix snapshots, a system prompt's head before any prompt's own
    /// and the least recently used first. A new prompt that cannot afford a head snapshot beside
    /// its own is prefilled the same way and keeps only its own. What the prompt continues is
    /// never dropped.
    ///
    /// **`Behavior`:** a dropped session is cold-prefilled when its conversation comes back,
    /// instead of reusing its KV (see `auto_session_enabled` for how those two differ), and a
    /// dropped snapshot is a prefix-cache miss. A request that does not reuse what was dropped
    /// gets the same text either way.
    pub(crate) memory_evict {
        env: "LUMEN_MLX_MEMORY_EVICT",
        default: true,
        kind: Behavior,
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
    turns: &[usize],
    label: &str,
) -> Result<(u32, usize, Option<usize>)> {
    let (seq_id, from, end) = (start.seq_id, start.from, prompt.len());
    let plan = match (boundary, runner.prefill_grid()) {
        (boundary, Some(grid)) if !turns.is_empty() => {
            session_feed::plan_turns(from, end, grid, boundary, turns)
        }
        (Some(b), Some(grid)) => session_feed::plan_feed(from, end, grid, b),
        _ => session_feed::FeedPlan {
            cuts: vec![end],
            mark: None,
        },
    };
    let t = Instant::now();
    let (mut last, mut pos, mut at) = (0, 0, from);
    let whole = match plan.mark {
        Some(mark) => runner.feed_marked(seq_id, from == 0, &prompt[from..], mark - from)?,
        None => None,
    };
    if let Some((l, p)) = whole {
        (last, pos) = (l, p);
    } else {
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
        SessionReuse::Restored => eprintln!(
            "[mlx] seq {seq_id} session={key:?}{label} restored from disk at {from} \
             suffix={fed} in {ms:.0}ms{marked}"
        ),
        SessionReuse::Fresh => eprintln!(
            "[mlx] seq {seq_id} session={key:?}{label} fresh prefill: {fed} tokens \
             in {ms:.0}ms{marked}"
        ),
    }
    Ok((last, pos, plan.mark.or(start.point)))
}

/// The tokens a turn leaves in its sequence: the prompt and the reply, up to `position`, the
/// number of tokens the sequence holds after the turn's last step.
///
/// A plain decode step feeds the token before the one it returns, so the last token a turn
/// returned is not among them. A session that listed it would start the next turn's `extend` one
/// token late, and the model would never see that token (most often the reply's end-of-turn
/// marker): the next turn would answer differently from a cold prefill of the same prompt.
/// Counting by the position rather than by "all but the last" also holds for a step that fed
/// more than it returned.
pub(crate) fn held_tokens(mut prompt: Vec<u32>, generated: &[u32], position: usize) -> Vec<u32> {
    prompt.extend_from_slice(generated);
    prompt.truncate(position);
    prompt
}

/// Write a turn's rollback point to the runner's disk tier, after the turn.
///
/// `prompt` is the turn's prompt, `point` the rollback point its sequence holds
/// (what [`feed`] returned) and `previous` the one it started from. The point
/// is keyed on the grid it is a cut of, so a later prompt restores it only on
/// that same grid.
///
/// A state is written when the point reaches a new chunk of the grid, not on
/// every turn: it holds the whole KV, and a point at a turn moves on each one.
/// After a restart the rest is re-fed, at most about a chunk.
///
/// Errors are logged, never returned: the reply is already out, and a missing
/// disk copy only costs a later prefill.
pub(crate) fn persist_point<R: SessionRunner + ?Sized>(
    runner: &mut R,
    key: &str,
    seq_id: u64,
    prompt: &[u32],
    point: Option<usize>,
    previous: Option<usize>,
    turn_cuts: bool,
) {
    let (Some(at), Some(grid)) = (point, runner.prefill_grid()) else {
        return;
    };
    if at == 0 || at > prompt.len() {
        return;
    }
    if grid.cut_at_or_before(at) == grid.cut_at_or_before(previous.unwrap_or(0)) {
        return;
    }
    let t = Instant::now();
    let grid_key = session_feed::point_grid_key(grid, at, turn_cuts);
    match runner.persist_point(seq_id, &prompt[..at], grid_key) {
        Ok(true) => eprintln!(
            "[mlx] seq {seq_id} session={key:?} persisted state at {at} to disk in {:.0}ms",
            t.elapsed().as_secs_f64() * 1000.0
        ),
        Ok(false) => {}
        Err(e) => eprintln!("[mlx] seq {seq_id} session={key:?} disk persist failed: {e:#}"),
    }
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
    ///
    /// A rollback point is usable only at a cut of this prompt's own cold
    /// prefill: a cut of the grid, or one of its `turns`.
    pub(crate) fn position<R: SessionRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        key: &str,
        prompt_ids: &[u32],
        turns: &[usize],
        label: &str,
        alloc_seq: impl FnOnce() -> u64,
    ) -> Result<SessionStart> {
        let grid = runner.prefill_grid();
        let is_cut =
            |at: usize| grid.is_some_and(|g| g.is_cut(at)) || turns.binary_search(&at).is_ok();
        let found = self.map.get(key).map(|s| {
            (
                s.seq_id,
                s.extends(prompt_ids).then_some(s.tokens.len()),
                s.resumes_at(prompt_ids)
                    .filter(|&kept| s.point_grid == grid && is_cut(kept)),
                s.rollback_len,
            )
        });
        let reused = found.and_then(|(_, extends, resumes, _)| extends.or(resumes));
        if reused.is_none() {
            // This key's session cannot serve the prompt and goes anyway: it is the first to
            // give its memory back.
            self.drop_divergent(runner, key, label);
        }
        let (kv_tokens, states) = feed_cost(reused.unwrap_or(0), prompt_ids.len());
        self.make_room(runner, key, kv_tokens, states);
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
        self.drop_divergent(runner, key, label);
        if reused.is_some() {
            // The rollback failed, so the whole prompt is fed, not the part after `reused`.
            let (kv_tokens, states) = feed_cost(0, prompt_ids.len());
            self.make_room(runner, key, kv_tokens, states);
        }
        let seq_id = alloc_seq();
        match runner.restore_point(seq_id, prompt_ids, turns) {
            Ok(Some(at)) => {
                return Ok(SessionStart {
                    seq_id,
                    from: at,
                    how: SessionReuse::Restored,
                    point: Some(at),
                });
            }
            Ok(None) => {}
            Err(e) => eprintln!(
                "[mlx] session={key:?}{label} disk restore failed ({e:#}); prefilling fresh"
            ),
        }
        Ok(SessionStart {
            seq_id,
            from: 0,
            how: SessionReuse::Fresh,
            point: None,
        })
    }

    /// Drop `key`'s session because its prompt diverged from it.
    fn drop_divergent<R: SessionRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        key: &str,
        label: &str,
    ) {
        if let Some(old) = self.map.remove(key) {
            let _ = runner.remove_seq(old.seq_id);
            eprintln!(
                "[mlx] session={key:?}{label} divergent prompt; dropping old seq {} ({} tokens)",
                old.seq_id,
                old.tokens.len()
            );
        }
    }

    /// Before a feed that will leave `kv_tokens` tokens of KV and `states` states in memory: drop
    /// the least recently used sessions other than `keep` for as long as the runner says it would
    /// go past its memory (see [`memory_evict`]). Stops when it fits, or when nothing else is
    /// left to drop, and then feeds anyway, as it always did. `keep` is `""` to spare none.
    pub(crate) fn make_room<R: SessionRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        keep: &str,
        kv_tokens: usize,
        states: usize,
    ) {
        if !memory_evict::get() {
            return;
        }
        loop {
            let short = runner.memory_shortfall(kv_tokens, states);
            if short == 0 {
                return;
            }
            let Some(victim) = self
                .map
                .iter()
                .filter(|(k, _)| k.as_str() != keep)
                .min_by_key(|(_, s)| s.last_access)
                .map(|(k, _)| k.clone())
            else {
                return;
            };
            if let Some(state) = self.map.remove(&victim) {
                let _ = runner.remove_seq(state.seq_id);
                eprintln!(
                    "[mlx] session={victim:?} evicted (memory: a feed of {kv_tokens} KV tokens \
                     would go {:.2} GB past what is planned); freed seq {} ({} tokens)",
                    short as f64 / 1e9,
                    state.seq_id,
                    state.tokens.len()
                );
            }
        }
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
                point_grid: runner.prefill_grid(),
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
/// A session's `tokens` are what the model was fed (its prompt and its reply up
/// to the last token), so a session whose tokens are a strict prefix of this prompt **is** this
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
            point_grid: None,
            last_access,
        }
    }

    fn session_with(tokens: Vec<u32>, last_access: Instant) -> SessionState {
        SessionState {
            seq_id: 0,
            tokens,
            rollback_len: None,
            point_grid: None,
            last_access,
        }
    }

    /// A stock OpenAI or Anthropic request carries no conversation id, and
    /// `session_id` is a Lumen extension no such client sends — so before this,
    /// every one of them re-prefilled the whole conversation, every turn.
    ///
    /// The prompt identifies its own conversation instead: a session's tokens
    /// are what the model was fed, its prompt and reply, so a strict prefix
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

    /// A turn's last returned token was never fed, so the session does not claim it, and the
    /// next prompt that repeats it extends the session from that token on.
    #[test]
    fn a_session_holds_the_reply_only_as_far_as_the_sequence_was_fed() {
        let (prompt, reply) = (vec![1, 2, 3], [7, 8, 9]);
        // Prefill fed 3 tokens, two decode steps fed 7 and 8 and returned 8 and 9.
        let held = held_tokens(prompt.clone(), &reply, 5);
        assert_eq!(held, [1, 2, 3, 7, 8]);
        let session = session_with(held, Instant::now());
        let next_turn = [1, 2, 3, 7, 8, 9, 4];
        assert_eq!(session.reusable_len(&next_turn), Some(5));
        // A reply that ended on its first token: only the prompt was fed.
        assert_eq!(held_tokens(prompt, &[7], 3), [1, 2, 3]);
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

    /// A runner with a memory budget: every sequence holds a fixed number of bytes and a feed
    /// adds `bytes_per_token` for each token.
    struct Budgeted {
        held: HashMap<u64, u64>,
        budget: u64,
        bytes_per_token: u64,
        dropped: Vec<u64>,
    }

    impl SessionRunner for Budgeted {
        fn prefill(&mut self, _: u64, _: &[u32]) -> Result<(u32, usize)> {
            unreachable!("only positioning is under test")
        }
        fn extend(&mut self, _: u64, _: &[u32]) -> Result<(u32, usize)> {
            unreachable!("only positioning is under test")
        }
        fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
            self.held.remove(&seq_id);
            self.dropped.push(seq_id);
            Ok(())
        }
        fn memory_shortfall(&self, kv_tokens: usize, _states: usize) -> u64 {
            let live: u64 = self.held.values().sum();
            (live + kv_tokens as u64 * self.bytes_per_token).saturating_sub(self.budget)
        }
    }

    /// Sessions `a` (the oldest) to `c` (the newest) of 100 bytes each, with seq ids 1 to 3.
    fn three_sessions(budget: u64) -> (Sessions, Budgeted) {
        let now = Instant::now();
        let mut sessions = Sessions::new();
        let mut runner = Budgeted {
            held: HashMap::new(),
            budget,
            bytes_per_token: 1,
            dropped: Vec::new(),
        };
        for (i, name) in ["a", "b", "c"].into_iter().enumerate() {
            let age = Duration::from_secs(30 - 10 * i as u64);
            sessions
                .map
                .insert(name.to_string(), session_with(vec![i as u32], now - age));
            sessions.map.get_mut(name).unwrap().seq_id = i as u64 + 1;
            runner.held.insert(i as u64 + 1, 100);
        }
        (sessions, runner)
    }

    fn keys(sessions: &Sessions) -> Vec<String> {
        let mut k: Vec<String> = sessions.map.keys().cloned().collect();
        k.sort();
        k
    }

    /// A prompt for a new conversation drops the sessions that do not fit beside it, the least
    /// recently used first, and stops as soon as it fits.
    #[test]
    fn a_new_prompt_drops_the_oldest_sessions_until_it_fits() {
        // 300 bytes held, 50 more for the prompt, room for 340: one session (the oldest) goes.
        let (mut sessions, mut runner) = three_sessions(340);
        sessions.make_room(&mut runner, "new", 50, 0);
        assert_eq!(runner.dropped, vec![1]);
        assert_eq!(keys(&sessions), vec!["b", "c"]);

        // Room for 240: two go, oldest first.
        let (mut sessions, mut runner) = three_sessions(240);
        sessions.make_room(&mut runner, "new", 50, 0);
        assert_eq!(runner.dropped, vec![1, 2]);
        assert_eq!(keys(&sessions), vec!["c"]);

        // It fits already: nothing is touched.
        let (mut sessions, mut runner) = three_sessions(1_000);
        sessions.make_room(&mut runner, "new", 50, 0);
        assert!(runner.dropped.is_empty());
        assert_eq!(keys(&sessions), vec!["a", "b", "c"]);
    }

    /// The conversation being continued is the one thing never dropped, however old, and a
    /// prompt that cannot fit even alone drops the rest and is still fed.
    #[test]
    fn the_session_being_continued_is_kept_and_a_prompt_too_big_for_the_budget_still_runs() {
        let (mut sessions, mut runner) = three_sessions(10);
        sessions.make_room(&mut runner, "a", 50, 0);
        assert_eq!(runner.dropped, vec![2, 3]);
        assert_eq!(keys(&sessions), vec!["a"]);
        // Nothing else left: the shortfall stays, and so does the session.
        sessions.make_room(&mut runner, "a", 50, 0);
        assert_eq!(keys(&sessions), vec!["a"]);
    }

    /// `position` makes the room before the sequence is allocated: a conversation that diverged
    /// from its own session gives that session up first, then the oldest others as needed.
    #[test]
    fn positioning_a_new_prompt_makes_room_first() {
        let (mut sessions, mut runner) = three_sessions(260);
        let start = sessions
            .position(&mut runner, "c", &[9, 9, 9], &[], "", || 4)
            .unwrap();
        assert_eq!(start.how, SessionReuse::Fresh);
        // "c" (seq 3) diverged and went at once; 200 held + 3 fits 260, so "a" and "b" stay.
        assert_eq!(runner.dropped, vec![3]);
        assert_eq!(keys(&sessions), vec!["a", "b"]);

        let (mut sessions, mut runner) = three_sessions(120);
        sessions
            .position(&mut runner, "new", &[9; 30], &[], "", || 4)
            .unwrap();
        assert_eq!(runner.dropped, vec![1, 2, 3]);
        assert!(sessions.map.is_empty());
    }

    #[test]
    fn the_memory_eviction_can_be_switched_off() {
        memory_evict::with(false, || {
            let (mut sessions, mut runner) = three_sessions(10);
            sessions.make_room(&mut runner, "new", 50, 0);
            assert!(runner.dropped.is_empty());
            assert_eq!(keys(&sessions), vec!["a", "b", "c"]);
        });
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

    /// Records the calls that built each sequence, carried through a rollback.
    #[derive(Default)]
    struct CutRunner {
        seqs: HashMap<u64, CutSeq>,
        fed: usize,
        persisted: Vec<(usize, u64)>,
    }

    #[derive(Default)]
    struct CutSeq {
        /// Where each call ended.
        calls: Vec<usize>,
        /// How many of `calls` the rollback point keeps.
        point: Option<usize>,
    }

    impl CutRunner {
        /// One call, split on the grid as the native runner's bulk pass splits it.
        fn call(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            let grid = self.prefill_grid().expect("a grid");
            let seq = self.seqs.entry(seq_id).or_default();
            let at = seq.calls.last().copied().unwrap_or(0);
            seq.calls.extend(grid.pieces(at, at + tokens.len()));
            self.fed += tokens.len();
            Ok((0, at + tokens.len()))
        }
    }

    impl SessionRunner for CutRunner {
        fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.call(seq_id, tokens)
        }
        fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.call(seq_id, tokens)
        }
        fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
            self.seqs.remove(&seq_id);
            Ok(())
        }
        fn prefill_grid(&self) -> Option<PrefillGrid> {
            Some(PrefillGrid::new(CHUNK, u64::MAX, 1))
        }
        fn mark_rollback(&mut self, seq_id: u64) -> Result<()> {
            let seq = self.seqs.get_mut(&seq_id).expect("a sequence to mark");
            seq.point = Some(seq.calls.len());
            Ok(())
        }
        fn rollback(&mut self, seq_id: u64) -> Result<usize> {
            let seq = self.seqs.get_mut(&seq_id).expect("a sequence to wind back");
            seq.calls.truncate(seq.point.expect("a point"));
            Ok(seq.calls.last().copied().unwrap_or(0))
        }
        fn persist_point(&mut self, _: u64, prefix: &[u32], grid_key: u64) -> Result<bool> {
            self.persisted.push((prefix.len(), grid_key));
            Ok(true)
        }
    }

    const CHUNK: usize = 64;
    /// How an assistant turn opens, replayed or generated.
    const OPEN: [u32; 3] = [900, 901, 902];
    /// What only a generation header has after [`OPEN`], as Qwen's empty think block.
    const HEADER_END: [u32; 2] = [903, 904];

    /// A plain chat of `rounds` finished rounds, as a generation prompt. Ids are distinct so a
    /// wrong prefix cannot pass; the reply of each round is replayed without [`HEADER_END`].
    fn chat(rounds: u32) -> (Vec<u32>, usize, Vec<usize>) {
        let mut prompt: Vec<u32> = (1..=20).collect();
        for r in 0..=rounds {
            prompt.extend((0..5 + r % 4).map(|i| 1000 + 100 * r + i));
            if r < rounds {
                prompt.extend(OPEN);
                prompt.extend((0..9 + r % 7).map(|i| 5000 + 100 * r + i));
            }
        }
        let boundary = prompt.len();
        prompt.extend(OPEN);
        prompt.extend(HEADER_END);
        let turns = prompt
            .windows(OPEN.len())
            .enumerate()
            .filter(|(_, w)| *w == OPEN)
            .map(|(i, _)| i)
            .collect();
        (prompt, boundary, turns)
    }

    struct Turn {
        seq_id: u64,
        how: SessionReuse,
        point: Option<usize>,
        started_at: Option<usize>,
    }

    /// One turn the way a family runs it: position, feed, and a reply of two tokens.
    fn chat_turn(sessions: &mut Sessions, runner: &mut CutRunner, rounds: u32, gap: usize) -> Turn {
        let (prompt, boundary, starts) = chat(rounds);
        let grid = runner.prefill_grid().expect("a grid");
        let turns = session_feed::kept_turns(&starts, grid, gap);
        let start = sessions
            .position(runner, "s", &prompt, &turns, "", || 100 + u64::from(rounds))
            .expect("position");
        let (_, pos, point) =
            feed(runner, "s", &start, &prompt, Some(boundary), &turns, "").expect("feed");
        let held = held_tokens(prompt, &[1, 2], pos + 1);
        sessions.store(runner, "s", start.seq_id, held, point);
        Turn {
            seq_id: start.seq_id,
            how: start.how,
            point,
            started_at: start.point,
        }
    }

    /// With turn cuts a warm turn is its own cold prefill, call for call, feeds only what follows
    /// the previous turn's point, and that point is never `gap` short of its boundary.
    #[test]
    fn a_warm_turn_is_its_cold_prefill_and_feeds_only_what_is_new() {
        for gap in [0, 30, 100] {
            let (mut sessions, mut warm) = (Sessions::new(), CutRunner::default());
            let mut previous = None;
            for rounds in 0..12 {
                let (prompt, boundary, _) = chat(rounds);
                let ctx = format!("gap {gap} round {rounds}");
                warm.fed = 0;
                let turn = chat_turn(&mut sessions, &mut warm, rounds, gap);

                let mut cold = CutRunner::default();
                let cold_turn = chat_turn(&mut Sessions::new(), &mut cold, rounds, gap);
                assert_eq!(
                    warm.seqs[&turn.seq_id].calls, cold.seqs[&cold_turn.seq_id].calls,
                    "{ctx}: a warm state from other calls than its cold prefill"
                );
                assert_eq!(turn.point, cold_turn.point, "{ctx}");
                assert!(boundary - turn.point.unwrap_or(0) < gap.max(1), "{ctx}");
                match previous {
                    Some(from) => {
                        assert_eq!(turn.how, SessionReuse::Resume, "{ctx}");
                        assert_eq!(warm.fed, prompt.len() - from, "{ctx}");
                    }
                    None => assert_eq!(turn.how, SessionReuse::Fresh, "{ctx}"),
                }
                previous = turn.point;
            }
        }
    }

    /// A point at a turn is no cut of a prompt that has no turns: `/v1/completions` prefills
    /// cold on the grid, so resuming there would not match it.
    #[test]
    fn a_turn_point_serves_only_a_prompt_cut_at_that_turn() {
        let (mut sessions, mut runner) = (Sessions::new(), CutRunner::default());
        let turn = chat_turn(&mut sessions, &mut runner, 2, 0);
        let at = turn.point.expect("a point");
        assert_ne!(at % CHUNK, 0, "pick a round whose boundary is off the grid");

        let (next, _, _) = chat(3);
        let start = sessions
            .position(&mut runner, "s", &next, &[], "", || 200)
            .expect("position");
        assert_eq!(start.how, SessionReuse::Fresh);
    }

    /// The state goes to disk once per chunk of the grid, whatever the number of turns, and
    /// under a key no state fed on the grid alone has.
    #[test]
    fn a_turn_point_is_written_once_per_chunk() {
        let (mut sessions, mut runner) = (Sessions::new(), CutRunner::default());
        let grid = runner.prefill_grid().expect("a grid");
        let mut expected = Vec::new();
        let mut last_chunk = 0;
        for rounds in 0..12 {
            let turn = chat_turn(&mut sessions, &mut runner, rounds, 0);
            let (prompt, _, _) = chat(rounds);
            persist_point(
                &mut runner,
                "s",
                turn.seq_id,
                &prompt,
                turn.point,
                turn.started_at,
                true,
            );
            let at = turn.point.expect("a point");
            if grid.cut_at_or_before(at) != last_chunk {
                last_chunk = grid.cut_at_or_before(at);
                expected.push((at, session_feed::point_grid_key(grid, at, true)));
            }
        }
        assert!(expected.len() > 2, "the chat must cross several chunks");
        assert!(expected.len() < 12, "and hold turns that share one");
        assert_eq!(runner.persisted, expected);
    }
}
