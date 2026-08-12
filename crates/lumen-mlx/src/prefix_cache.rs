//! Track A1 prefix-cache machinery — extracted backend-agnostic.
//!
//! A [`PrefixCacheStore`] holds master KV snapshots taken at the end of a
//! *shared* prompt prefix (typically the system-prompt block), keyed by a
//! process-stable hash. [`PrefixCacheStore::prefill_optionally_cached`] is the
//! HIT / MISS / exact-retry driver: on HIT it forks the stored snapshot and
//! extends only the suffix; on MISS it cold-prefills and snapshots the result
//! so the next request can fork from it.
//!
//! **Multi-entry LPM ("tree-lite", WS-A lever #1).** Each key maps to a *small
//! `Vec`* of candidate prefix snapshots rather than a single linear chain. A
//! branching agentic session that shares a system + early turns and then
//! diverges keeps one entry per divergent tail, and lookup selects the
//! longest-common-prefix (LPM) candidate to fork from. On a partial match that
//! diverges (shared head, different tail) a NEW entry is branched rather than
//! overwriting the existing one, so both tails stay warm. Per-key branch count
//! is capped (`LUMEN_MLX_PREFIX_CACHE_BRANCHES`, default 4) and total entries
//! across keys is bounded by the LRU/TTL policy. An entry currently being
//! forked / extended is `in_use`-locked and never evicted mid-flight.
//!
//! When a key holds exactly one entry (the common single-conversation case) the
//! LPM degenerates to "pick the one candidate", so the fork / extend / snapshot
//! sequence — and therefore the returned token and KV state — is bit-identical
//! to the previous single-entry implementation.
//!
//! The driver is generic over [`SnapshotRunner`], so any backend whose runner
//! can snapshot / fork / extend its KV state reuses this logic. It is currently
//! wired into the Qwen3.6 tools path (non-rotating `NativeKvCache`). Gemma 4 —
//! which uses a rotating / sliding-window cache where a snapshot taken before
//! the window slides past it can no longer be rolled back to — can adopt the
//! store + TTL/LRU here, but needs a boundary-snapshot fix before its forks
//! survive window rotation (see `gemma4-prefix-cache-broken` in project memory).

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default per-key branch cap when `LUMEN_MLX_PREFIX_CACHE_BRANCHES` is unset.
const DEFAULT_BRANCHES: usize = 4;

/// Default for incremental chunked-prefill boundary caching (Phase 0). **ON**:
/// on a cold MISS the stable head (system + rendered tools, via the tools-aware
/// boundary in `lib.rs::detect_system_tools_prefix_len*`) is snapshotted as a
/// `pinned_boundary` branch so a later same-system/same-tools but divergent-tail
/// prompt (e.g. a client compaction/resume) FORKS the ~48K head instead of
/// cold-prefilling it again (~98s → ~1-2s on the agentic path). Set
/// `LUMEN_MLX_PREFIX_INCREMENTAL=0` to revert to the byte-identical single-prefill.
const DEFAULT_INCREMENTAL: bool = true;

/// The minimal runner surface the prefix cache needs. A backend implements this
/// for its runner type — e.g. a blanket impl over a richer internal `Runner`
/// trait — so the cache stays decoupled from any one runner enum.
pub trait SnapshotRunner {
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    /// Deep-copy snapshot for fork-to-new-seq. Returns `(snapshot_id, position)`.
    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)>;
    /// Seed a fresh `dst_seq_id` from a deep snapshot (snapshot is *not*
    /// consumed — multi-fork supported). Returns the new seq's position.
    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize>;
    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()>;

    /// L2 disk tier (opt-in): durably persist the deep snapshot `snapshot_id`
    /// under `key` so it survives process restart / in-memory eviction. Default
    /// is a no-op, so in-memory-only runners (and the disabled disk path) are
    /// byte-identical to before. Errors are non-fatal — the caller logs and
    /// continues without the durable copy.
    fn persist_snapshot(
        &mut self,
        _snapshot_id: u64,
        _key: &str,
        _prefix_tokens: &[u32],
        _last_token: Option<u32>,
    ) -> Result<()> {
        Ok(())
    }

    /// L2 disk tier (opt-in): try to rehydrate a previously persisted snapshot
    /// for `key` whose token prefix is a prefix of `prompt_ids`. On success the
    /// runner materializes the snapshot into its in-memory snapshot table and
    /// returns `(snapshot_id, prefix_tokens, last_token)` so the store can
    /// register an entry and fork from it. Default returns `None` (no disk
    /// tier), making the cold-MISS path unchanged.
    fn load_persisted_snapshot(
        &mut self,
        _key: &str,
        _prompt_ids: &[u32],
    ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
        Ok(None)
    }
}

/// A prefix-cache master snapshot entry. Holds a deep snapshot of the per-layer
/// cache state taken at the end of a *shared* prefix (typically the system-
/// prompt block). Multiple incoming requests whose prompt starts with
/// `prefix_tokens` reuse this snapshot via [`SnapshotRunner::fork_from_snapshot`],
/// paying only the suffix prefill instead of the full prefix prefill.
///
/// Master is reusable; `release_snapshot` is only called on cache eviction
/// (LRU / TTL / explicit drop).
pub struct PrefixCacheEntry {
    pub master_snapshot_id: u64,
    pub prefix_tokens: Vec<u32>,
    pub last_access: Instant,
    pub hits: u64,
    /// Argmax token predicted at the end of `prefix_tokens` (i.e. the first
    /// decode token for a prompt equal to `prefix_tokens`). `Some` only when the
    /// snapshot was taken at the *full* prompt end with logits available (the
    /// tools cold-prefill / advance paths). `None` for boundary snapshots
    /// (streaming path), which always re-extend a trailing header to obtain
    /// fresh logits and so never need this. Lets an identical-prompt retry HIT
    /// (empty suffix) reuse the completed snapshot instead of re-prefilling.
    pub last_token: Option<u32>,
    /// In-flight fork/extend guard. While `> 0` this entry's snapshot is being
    /// forked or extended from, so eviction must skip it: releasing the snapshot
    /// mid-fork would corrupt the in-progress request. Bumped around fork+extend
    /// and decremented when the operation completes.
    pub in_use: u32,
    /// Set on an *incremental boundary* branch (a snapshot of the shared
    /// system-prompt head taken mid-cold-prefill, Phase 0). Such a branch must
    /// survive when the full-prompt branch is inserted afterwards: the full
    /// prompt `starts_with` the head, so `insert_branch`'s supersede path would
    /// otherwise release the head and silently defeat incremental caching. The
    /// guard keeps both so a later prefix-sharing-but-divergent prompt can still
    /// fork the head. Always `false` for full-prompt / boundary-store entries.
    pub pinned_boundary: bool,
}

/// Keyed store of prefix-cache master snapshots plus its TTL / LRU policy.
///
/// Each key maps to a `Vec` of candidate entries (branches). Lookup selects the
/// longest-common-prefix candidate; divergence pushes a new branch up to the
/// per-key cap.
pub struct PrefixCacheStore {
    entries: HashMap<String, Vec<PrefixCacheEntry>>,
    enabled: bool,
    ttl: Option<Duration>,
    max: Option<usize>,
    /// Max candidate branches kept per key.
    branches: usize,
    /// Incremental chunked-prefill boundary caching (Phase 0). When ON, a cold
    /// MISS with a supplied shared-prefix boundary snapshots the head as a
    /// `pinned_boundary` branch before extending the tail, so a later
    /// prefix-sharing-but-divergent prompt forks the head instead of cold-
    /// prefilling. Default OFF (bit-identical desktop behaviour).
    incremental: bool,
}

impl PrefixCacheStore {
    /// Construct from the `LUMEN_MLX_PREFIX_CACHE*` env vars.
    pub fn from_env() -> Self {
        let (enabled, ttl, max, branches, incremental) = read_limits();
        Self {
            entries: HashMap::new(),
            enabled,
            ttl,
            max,
            branches,
            incremental,
        }
    }

    /// Construct with an explicit policy (test-only — production uses
    /// [`PrefixCacheStore::from_env`]).
    #[cfg(test)]
    fn with_policy(enabled: bool, ttl: Option<Duration>, max: Option<usize>) -> Self {
        Self {
            entries: HashMap::new(),
            enabled,
            ttl,
            max,
            branches: DEFAULT_BRANCHES,
            incremental: false,
        }
    }

    /// Construct with explicit policy + branch cap (test-only).
    #[cfg(test)]
    fn with_branches(
        enabled: bool,
        ttl: Option<Duration>,
        max: Option<usize>,
        branches: usize,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            enabled,
            ttl,
            max,
            branches: branches.max(1),
            incremental: false,
        }
    }

    /// Construct with explicit policy + branch cap + incremental flag (test-only).
    #[cfg(test)]
    fn with_incremental(enabled: bool, branches: usize, incremental: bool) -> Self {
        Self {
            entries: HashMap::new(),
            enabled,
            ttl: None,
            max: None,
            branches: branches.max(1),
            incremental,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    pub fn max(&self) -> Option<usize> {
        self.max
    }

    /// Prefill `prompt_ids` for `seq_id`, consulting the store when `key` is
    /// `Some` and the feature is enabled.
    ///
    /// On HIT (the new prompt extends — or is byte-identical to — a cached
    /// prefix), forks the longest-matching candidate snapshot and extends by the
    /// suffix, skipping re-processing the common prefix. On MISS, runs a normal
    /// prefill and snapshots the result under `key`; if the new prompt shares a
    /// non-trivial head with an existing candidate but diverges in the tail, the
    /// result is stored as a NEW branch instead of replacing the divergent one.
    ///
    /// Falls through to a plain `runner.prefill` when `key` is `None` or the
    /// feature is disabled. Snapshot failures are logged but non-fatal — the
    /// request completes, just without (re)populating the cache.
    ///
    /// `incremental_boundary` (Phase 0) is `Some(sys_len)` when the caller knows
    /// the length, in tokens, of a shared prefix (the system-prompt block) that
    /// later divergent requests are likely to share. On a cold MISS with the
    /// `incremental` policy enabled and a valid boundary (`0 < b < len`), the
    /// head `prompt_ids[..b]` is snapshotted as a `pinned_boundary` branch before
    /// the tail is extended — so a future `[sys + differentTail]` request forks
    /// the head instead of cold-prefilling. `None` (or feature off / invalid
    /// boundary) falls back to the exact single-prefill MISS path.
    pub fn prefill_optionally_cached<R: SnapshotRunner>(
        &mut self,
        runner: &mut R,
        seq_id: u64,
        prompt_ids: &[u32],
        key: Option<&str>,
        incremental_boundary: Option<usize>,
    ) -> Result<(u32, usize)> {
        let key = match key.filter(|_| self.enabled) {
            Some(k) => k,
            None => return runner.prefill(seq_id, prompt_ids),
        };
        self.evict_stale(runner);

        // LPM: among this key's candidates, pick the one whose prefix is the
        // longest prefix of `prompt_ids`. HIT requires either a strict extension
        // (suffix non-empty) or a byte-identical prompt with a stored argmax
        // (`last_token`) — the omp exact-retry case.
        let hit = match self.longest_hit(key, prompt_ids) {
            Some(idx) => Some(idx),
            // L2 disk tier: on an in-memory miss, try to rehydrate a persisted
            // snapshot for this key (survives restart / eviction). With the disk
            // tier off, `load_persisted_snapshot` defaults to `None`, so this is
            // byte-identical to the original cold-MISS path. On a disk HIT we
            // register the rehydrated snapshot as a branch and re-run LPM so the
            // standard fork+extend path below handles it uniformly.
            None => match runner.load_persisted_snapshot(key, prompt_ids) {
                Ok(Some((snap_id, prefix_tokens, last_token))) => {
                    self.insert_branch(
                        runner,
                        key,
                        PrefixCacheEntry {
                            master_snapshot_id: snap_id,
                            prefix_tokens,
                            last_access: Instant::now(),
                            hits: 0,
                            last_token,
                            in_use: 0,
                            pinned_boundary: false,
                        },
                    );
                    self.longest_hit(key, prompt_ids)
                }
                Ok(None) => None,
                Err(e) => {
                    eprintln!("[mlx] prefix-cache: disk rehydrate failed (key={key:?}): {e:#}");
                    None
                }
            },
        };

        if let Some(idx) = hit {
            let (master, prefix, cached_last) = {
                let e = &self.entries[key][idx];
                (e.master_snapshot_id, e.prefix_tokens.clone(), e.last_token)
            };
            let suffix = &prompt_ids[prefix.len()..];
            let t = Instant::now();
            // Lock the source entry across fork+extend so a re-entrant eviction
            // cannot release `master` mid-flight.
            self.set_in_use(key, idx, true);
            let fork_res = runner.fork_from_snapshot(master, seq_id);
            let (last, pos) = match fork_res {
                Ok(_) => {
                    if suffix.is_empty() {
                        // Exact-prompt retry: the snapshot's KV already covers the
                        // whole prompt, nothing to extend. `cached_last` is `Some`
                        // here (guaranteed by `longest_hit`), and equals what a
                        // cold prefill of this prompt would have returned.
                        (
                            cached_last.expect("longest_hit guarantees Some on empty suffix"),
                            prefix.len(),
                        )
                    } else {
                        match runner.extend(seq_id, suffix) {
                            Ok(le) => le,
                            Err(e) => {
                                self.set_in_use(key, idx, false);
                                return Err(e);
                            }
                        }
                    }
                }
                Err(e) => {
                    self.set_in_use(key, idx, false);
                    return Err(e);
                }
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            // ADVANCE the matched candidate to THIS turn's full prompt so the
            // NEXT turn forks from here and re-prefills only its own new tokens.
            // Without this the snapshot stays frozen at the first turn's prompt
            // and the reused `prefix` never grows — so every turn re-prefills the
            // whole (growing) conversation suffix. The trailing generation header
            // is a clean prefix of the next turn because a completed assistant
            // turn re-renders with the same `<think>\n\n</think>\n\n` block.
            // Skipped on an empty suffix: the forked state already equals the
            // stored snapshot, so re-snapshotting would only churn ~3GB of KV.
            let mut advanced_to = prefix.len();
            if suffix.is_empty() {
                self.set_in_use(key, idx, false);
                if let Some(e) = self.entries.get_mut(key).and_then(|v| v.get_mut(idx)) {
                    e.last_access = Instant::now();
                    e.hits += 1;
                }
            } else {
                let matched_pinned = self
                    .entries
                    .get(key)
                    .and_then(|v| v.get(idx))
                    .is_some_and(|e| e.pinned_boundary);
                match runner.snapshot_state_deep(seq_id) {
                    Ok((snap_id, _)) if matched_pinned => {
                        // The match was an incremental boundary head (Phase 0).
                        // KEEP it so future divergent prefix-sharing prompts can
                        // still fork it, and add the advanced full prompt as a
                        // SEPARATE branch. (Replacing it in place would collapse
                        // the shared boundary after the first divergent hit.)
                        self.set_in_use(key, idx, false);
                        if let Some(e) = self.entries.get_mut(key).and_then(|v| v.get_mut(idx)) {
                            e.last_access = Instant::now();
                            e.hits += 1;
                        }
                        self.insert_branch(
                            runner,
                            key,
                            PrefixCacheEntry {
                                master_snapshot_id: snap_id,
                                prefix_tokens: prompt_ids.to_vec(),
                                last_access: Instant::now(),
                                hits: 0,
                                last_token: Some(last),
                                in_use: 0,
                                pinned_boundary: false,
                            },
                        );
                        self.evict_stale(runner);
                        advanced_to = prompt_ids.len();
                    }
                    Ok((snap_id, _)) => {
                        // Replace the matched candidate in place with the advanced
                        // snapshot, releasing the superseded one. (Same key/index
                        // → single-entry behaviour is identical.)
                        let old = {
                            let e = &mut self.entries.get_mut(key).unwrap()[idx];
                            let prev = e.master_snapshot_id;
                            let hits = e.hits + 1;
                            *e = PrefixCacheEntry {
                                master_snapshot_id: snap_id,
                                prefix_tokens: prompt_ids.to_vec(),
                                last_access: Instant::now(),
                                hits,
                                last_token: Some(last),
                                in_use: 0,
                                pinned_boundary: false,
                            };
                            prev
                        };
                        let _ = runner.release_snapshot(old);
                        self.evict_stale(runner);
                        advanced_to = prompt_ids.len();
                    }
                    Err(e) => {
                        self.set_in_use(key, idx, false);
                        if let Some(en) = self.entries.get_mut(key).and_then(|v| v.get_mut(idx)) {
                            en.last_access = Instant::now();
                            en.hits += 1;
                        }
                        eprintln!("[mlx] prefix-cache: HIT snapshot-advance failed ({e:#})");
                    }
                }
            }
            eprintln!(
                "[mlx] prefix-cache HIT (tools) key={key:?} prefix={} suffix={} \
                 fork+extend={ms:.0}ms (advanced→{advanced_to})",
                prefix.len(),
                suffix.len()
            );
            return Ok((last, pos));
        }

        // Cold prefill, then snapshot under `key`. If the prompt shares a
        // non-trivial head with an existing branch but diverges in the tail,
        // store it as a NEW branch rather than overwriting the divergent one.
        //
        // Phase 0 incremental boundary caching: when enabled and the caller
        // supplied a valid shared-prefix boundary, cold-prefill the head ONLY,
        // snapshot it as a `pinned_boundary` branch, then extend the divergent
        // tail. `extend` after `prefill` produces a cache byte-identical to a
        // single `prefill` of the concatenation (same `forward_chunked`
        // contract the HIT path relies on), so the returned `(last, pos)` is
        // unchanged from a one-shot prefill — only an extra reusable snapshot is
        // created. With the feature off (or no/invalid boundary) this is the
        // exact original single-prefill path.
        let do_incremental =
            self.incremental && incremental_boundary.is_some_and(|b| b > 0 && b < prompt_ids.len());
        let (last, pos) = if do_incremental {
            let b = incremental_boundary.unwrap();
            let _ = runner.prefill(seq_id, &prompt_ids[..b])?;
            match runner.snapshot_state_deep(seq_id) {
                Ok((snap_b, _)) => {
                    // last_token = None: an exact-prompt-==-head retry must
                    // re-extend for fresh logits (same contract as the streaming
                    // boundary `store_master`). `pinned_boundary` keeps it alive
                    // when the full-prompt branch is inserted below.
                    self.insert_branch(
                        runner,
                        key,
                        PrefixCacheEntry {
                            master_snapshot_id: snap_b,
                            prefix_tokens: prompt_ids[..b].to_vec(),
                            last_access: Instant::now(),
                            hits: 0,
                            last_token: None,
                            in_use: 0,
                            pinned_boundary: true,
                        },
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[mlx] prefix-cache: incremental boundary snapshot skipped \
                         (key={key:?}, boundary={b}): {e:#}"
                    );
                }
            }
            runner.extend(seq_id, &prompt_ids[b..])?
        } else {
            runner.prefill(seq_id, prompt_ids)?
        };
        match runner.snapshot_state_deep(seq_id) {
            Ok((snap_id, _snap_pos)) => {
                self.insert_branch(
                    runner,
                    key,
                    PrefixCacheEntry {
                        master_snapshot_id: snap_id,
                        prefix_tokens: prompt_ids.to_vec(),
                        last_access: Instant::now(),
                        hits: 0,
                        last_token: Some(last),
                        in_use: 0,
                        pinned_boundary: false,
                    },
                );
                self.evict_stale(runner);
                // L2 disk tier (opt-in): durably persist the freshly-prefilled
                // snapshot so a future process can fork it instead of paying the
                // cold prefill again. No-op when the disk tier is off.
                if let Err(e) = runner.persist_snapshot(snap_id, key, prompt_ids, Some(last)) {
                    eprintln!("[mlx] prefix-cache: disk persist skipped (key={key:?}): {e:#}");
                }
                eprintln!(
                    "[mlx] prefix-cache MISS (tools) key={key:?} stored snapshot={snap_id} \
                     prefix_len={} branches={}",
                    prompt_ids.len(),
                    self.entries.get(key).map(|v| v.len()).unwrap_or(0),
                );
            }
            Err(e) => {
                eprintln!("[mlx] prefix-cache snapshot skipped (tools, key={key:?}): {e:#}");
            }
        }
        Ok((last, pos))
    }

    /// Select the index of the candidate under `key` whose `prefix_tokens` is the
    /// longest prefix of `prompt_ids` and qualifies as a HIT. A candidate
    /// qualifies when its prefix is non-empty and a prefix of `prompt_ids`, and
    /// either the suffix is non-empty (strict extension) or `last_token` is set
    /// (exact-retry). Ties break toward the longer prefix; equal length keeps the
    /// first (deterministic).
    fn longest_hit(&self, key: &str, prompt_ids: &[u32]) -> Option<usize> {
        let cands = self.entries.get(key)?;
        let mut best: Option<(usize, usize)> = None; // (index, prefix_len)
        for (i, e) in cands.iter().enumerate() {
            let p = &e.prefix_tokens;
            if p.is_empty() || !prompt_ids.starts_with(p) {
                continue;
            }
            let extends = prompt_ids.len() > p.len();
            if !(extends || e.last_token.is_some()) {
                continue;
            }
            match best {
                Some((_, bl)) if bl >= p.len() => {}
                _ => best = Some((i, p.len())),
            }
        }
        best.map(|(i, _)| i)
    }

    /// Length of the longest common token prefix between `a` and `b`.
    fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Insert `entry` as a branch under `key`. If an existing candidate is a
    /// prefix-of / equal-to the new prompt (so the new one strictly supersedes
    /// it along the same line), that candidate is replaced in place and its
    /// snapshot released — preserving single-entry identity. Otherwise the entry
    /// is pushed as a new branch, evicting the per-key least-recent candidate
    /// when the branch cap is exceeded.
    fn insert_branch<R: SnapshotRunner>(
        &mut self,
        runner: &mut R,
        key: &str,
        entry: PrefixCacheEntry,
    ) {
        let cands = self.entries.entry(key.to_string()).or_default();

        // Supersede an existing candidate that lies on the same line: either the
        // existing prefix is a prefix of the new one (new extends old), or they
        // are byte-identical. Never supersede an `in_use` candidate, nor a
        // `pinned_boundary` head (Phase 0): the full prompt always `starts_with`
        // the shared head, so without this guard inserting the full-prompt branch
        // would release the incremental boundary snapshot and defeat the feature.
        let supersede = cands.iter().position(|e| {
            e.in_use == 0
                && !e.pinned_boundary
                && !e.prefix_tokens.is_empty()
                && entry.prefix_tokens.starts_with(&e.prefix_tokens)
        });
        if let Some(i) = supersede {
            let old = std::mem::replace(&mut cands[i], entry);
            let _ = runner.release_snapshot(old.master_snapshot_id);
            return;
        }

        cands.push(entry);

        // Enforce the per-key branch cap: drop the least-recently-used candidate
        // that is not in use. (Total-entry LRU is handled separately by
        // `evict_stale`.)
        while cands.len() > self.branches {
            let victim = cands
                .iter()
                .enumerate()
                .filter(|(_, e)| e.in_use == 0)
                .min_by_key(|(_, e)| e.last_access)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let old = cands.remove(i);
                    let _ = runner.release_snapshot(old.master_snapshot_id);
                }
                // Every candidate is locked — keep them all rather than corrupt
                // an in-flight fork; the cap is soft under contention.
                None => break,
            }
        }
    }

    /// Bump / clear the in-use guard on a specific candidate.
    fn set_in_use(&mut self, key: &str, idx: usize, on: bool) {
        if let Some(e) = self.entries.get_mut(key).and_then(|v| v.get_mut(idx)) {
            if on {
                e.in_use += 1;
            } else {
                e.in_use = e.in_use.saturating_sub(1);
            }
        }
    }

    /// Drop entries exceeding the TTL / LRU policy, releasing their snapshots.
    /// In-use candidates are never evicted. No-op when both TTL and max are unset.
    pub fn evict_stale<R: SnapshotRunner>(&mut self, runner: &mut R) {
        let now = Instant::now();
        let victims = pick_eviction_victims(&self.entries, now, self.ttl, self.max);
        for (key, idx, reason) in victims {
            if let Some(cands) = self.entries.get_mut(&key) {
                if idx < cands.len() {
                    let entry = cands.remove(idx);
                    let _ = runner.release_snapshot(entry.master_snapshot_id);
                    eprintln!("[mlx] prefix-cache key={key:?} branch evicted ({reason})");
                }
                if cands.is_empty() {
                    self.entries.remove(&key);
                }
            }
        }
    }

    /// L2 spill: drop the coldest in-memory snapshots (releasing them from the
    /// runner) until at most `keep_n` remain. Spilled entries' KV stays on the
    /// disk tier (persisted at store time) and rehydrates on next access, so
    /// this frees RAM under memory pressure without losing the cache. In-use
    /// entries are never spilled. No-op when already at/below `keep_n`.
    pub fn spill<R: SnapshotRunner>(&mut self, runner: &mut R, keep_n: usize) {
        while self.len() > keep_n {
            let victim = self
                .entries
                .iter()
                .flat_map(|(k, v)| {
                    v.iter()
                        .enumerate()
                        .filter(|(_, e)| e.in_use == 0)
                        .map(move |(i, e)| (k.clone(), i, e.last_access))
                })
                .min_by_key(|(_, _, la)| *la)
                .map(|(k, i, _)| (k, i));
            match victim {
                Some((k, i)) => {
                    if let Some(cands) = self.entries.get_mut(&k) {
                        if i < cands.len() {
                            let e = cands.remove(i);
                            let _ = runner.release_snapshot(e.master_snapshot_id);
                        }
                        if cands.is_empty() {
                            self.entries.remove(&k);
                        }
                    }
                }
                // Every remaining entry is in-use — stop rather than spin.
                None => break,
            }
        }
    }

    /// Look up the master snapshot for `key` whose prefix is the longest prefix
    /// of `prompt_ids` (LPM), returning `(snapshot_id, prefix tokens)`. Used by
    /// the streaming boundary path to decide fork-vs-cold. With a single
    /// candidate this is exactly "the one entry".
    pub fn get_master_for(&self, key: &str, prompt_ids: &[u32]) -> Option<(u64, Vec<u32>)> {
        let cands = self.entries.get(key)?;
        cands
            .iter()
            .filter(|e| !e.prefix_tokens.is_empty() && prompt_ids.starts_with(&e.prefix_tokens))
            .max_by_key(|e| e.prefix_tokens.len())
            .map(|e| (e.master_snapshot_id, e.prefix_tokens.clone()))
    }

    /// Store a fresh master snapshot for `key` at the given `prefix_tokens`,
    /// branching on divergence. Used by the streaming boundary path (which passes
    /// `last_token = None`). A candidate on the same line is superseded; an
    /// off-line prompt becomes a new branch (capped). Then enforces TTL / LRU.
    pub fn store_master<R: SnapshotRunner>(
        &mut self,
        runner: &mut R,
        key: &str,
        snapshot_id: u64,
        prefix_tokens: Vec<u32>,
        last_token: Option<u32>,
    ) {
        self.insert_branch(
            runner,
            key,
            PrefixCacheEntry {
                master_snapshot_id: snapshot_id,
                prefix_tokens,
                last_access: Instant::now(),
                hits: 0,
                last_token,
                in_use: 0,
                pinned_boundary: false,
            },
        );
        self.evict_stale(runner);
    }

    /// Drop all branches under `key`, releasing their snapshots. Returns the
    /// summed `hits` count if the key existed.
    pub fn drop_entry<R: SnapshotRunner>(&mut self, runner: &mut R, key: &str) -> Option<u64> {
        let cands = self.entries.remove(key)?;
        let mut hits = 0;
        for entry in cands {
            hits += entry.hits;
            let _ = runner.release_snapshot(entry.master_snapshot_id);
        }
        Some(hits)
    }

    /// Drop all entries, releasing every snapshot. Returns the number of
    /// snapshots released.
    pub fn clear<R: SnapshotRunner>(&mut self, runner: &mut R) -> usize {
        let mut n = 0;
        let entries: Vec<_> = self.entries.drain().collect();
        for (_, cands) in entries {
            for entry in cands {
                n += 1;
                let _ = runner.release_snapshot(entry.master_snapshot_id);
            }
        }
        n
    }

    /// Total number of cached snapshots across all keys and branches.
    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }
}

/// Read A1 prefix-cache config from env.
///
/// - `LUMEN_MLX_PREFIX_CACHE=0`            — opt OUT (default ON since v0.4.7).
/// - `LUMEN_MLX_PREFIX_CACHE_TTL_SECS=N`   — drop entries idle > N seconds.
/// - `LUMEN_MLX_PREFIX_CACHE_MAX=N`        — keep at most N entries, LRU-evict.
/// - `LUMEN_MLX_PREFIX_CACHE_BRANCHES=N`   — keep at most N branches per key
///   (default 4). 1 restores strict single-entry-per-key behaviour.
/// - `LUMEN_MLX_PREFIX_INCREMENTAL=1`      — Phase 0: on a cold MISS, also
///   snapshot the shared system-prefix head as a pinned branch so a later
///   prefix-sharing-but-divergent prompt forks the head instead of cold-
///   prefilling. Default OFF (cold-MISS path stays byte-identical).
///
/// Default is ON: the feature has been validated since 2026-05-18 and provides
/// the 5-6× speedup users expect for repeated chat turns with a shared system
/// prompt. The env var is the escape hatch for debugging, not the gate.
fn read_limits() -> (bool, Option<Duration>, Option<usize>, usize, bool) {
    // Default ON: anything other than an explicit "0" / "false" / "no" / "off"
    // stays enabled. Empty string + unset both yield ON.
    let enabled = match std::env::var("LUMEN_MLX_PREFIX_CACHE")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
    {
        None => true,
        Some(v) if v.is_empty() => true,
        Some(v) => !matches!(v.as_str(), "0" | "false" | "no" | "off"),
    };
    let ttl = std::env::var("LUMEN_MLX_PREFIX_CACHE_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(Duration::from_secs);
    let max = std::env::var("LUMEN_MLX_PREFIX_CACHE_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let branches = std::env::var("LUMEN_MLX_PREFIX_CACHE_BRANCHES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_BRANCHES);
    // Incremental boundary caching is opt-IN (default OFF): only an explicit
    // truthy value turns it on, mirroring (inverted) the cache's default-ON gate.
    let incremental = match std::env::var("LUMEN_MLX_PREFIX_INCREMENTAL")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
    {
        Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
        None => DEFAULT_INCREMENTAL,
    };
    (enabled, ttl, max, branches, incremental)
}

/// Pure TTL-then-LRU victim selection across the multi-entry store. Returns
/// `(key, branch index, reason)` triples to evict, sorted so that evicting them
/// in order (with later indices first within each key, which the caller does via
/// `Vec::remove` per descending index) stays correct. In-use candidates are
/// never selected. Kept side-effect-free for unit testing; the caller releases
/// snapshots.
fn pick_eviction_victims(
    entries: &HashMap<String, Vec<PrefixCacheEntry>>,
    now: Instant,
    ttl: Option<Duration>,
    max: Option<usize>,
) -> Vec<(String, usize, String)> {
    // Flatten to (key, idx, last_access, in_use), the candidate universe.
    #[derive(Clone)]
    struct Cand {
        key: String,
        idx: usize,
        last_access: Instant,
        locked: bool,
    }
    let mut alive: Vec<Cand> = entries
        .iter()
        .flat_map(|(k, v)| {
            v.iter().enumerate().map(move |(i, e)| Cand {
                key: k.clone(),
                idx: i,
                last_access: e.last_access,
                locked: e.in_use > 0,
            })
        })
        .collect();

    let mut out: Vec<(String, usize, String)> = Vec::new();

    if let Some(ttl) = ttl {
        let stale: Vec<usize> = alive
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.locked && now.duration_since(c.last_access) > ttl)
            .map(|(i, _)| i)
            .collect();
        // Mark for removal; record victims.
        for &i in &stale {
            out.push((
                alive[i].key.clone(),
                alive[i].idx,
                format!("TTL > {}s", ttl.as_secs()),
            ));
        }
        // Drop staled from the alive set (descending index to keep positions).
        for &i in stale.iter().rev() {
            alive.remove(i);
        }
    }

    if let Some(max) = max {
        // Count only unlocked toward the budget pressure but never evict locked.
        while alive.iter().filter(|c| !c.locked).count() > max {
            let victim = alive
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.locked)
                .min_by_key(|(_, c)| c.last_access)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    out.push((alive[i].key.clone(), alive[i].idx, format!("LRU cap {max}")));
                    alive.remove(i);
                }
                None => break,
            }
        }
    }

    // Critical: when several victims share a key, removing a lower index first
    // shifts higher indices. Sort so that within each key we remove HIGHER
    // indices first. Group-stable: sort by (key, descending idx).
    out.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the exact runner calls the driver makes so tests can assert the
    /// HIT / MISS / exact-retry control flow without a real model. `prefill`
    /// and `extend` return distinct sentinel argmax tokens so the returned
    /// token pins down which branch produced it.
    #[derive(Default)]
    struct MockRunner {
        log: Vec<String>,
        next_snap: u64,
    }
    const PREFILL_TOK: u32 = 100;
    const EXTEND_TOK: u32 = 200;

    impl SnapshotRunner for MockRunner {
        fn prefill(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.log.push(format!("prefill({})", tokens.len()));
            Ok((PREFILL_TOK, tokens.len()))
        }
        fn extend(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.log.push(format!("extend({})", tokens.len()));
            Ok((EXTEND_TOK, tokens.len()))
        }
        fn snapshot_state_deep(&mut self, _seq: u64) -> Result<(u64, usize)> {
            let id = self.next_snap;
            self.next_snap += 1;
            self.log.push(format!("snapshot->{id}"));
            Ok((id, 0))
        }
        fn fork_from_snapshot(&mut self, snapshot_id: u64, _dst: u64) -> Result<usize> {
            self.log.push(format!("fork({snapshot_id})"));
            Ok(0)
        }
        fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
            self.log.push(format!("release({snapshot_id})"));
            Ok(())
        }
    }

    // ───────────────── interrupt integrity (005 Phase 3.1) ─────────────────
    //
    // SQLite's crash testing asserts "committed or completely rolled back" of
    // durable state. lumen has almost none, but it has the same property to
    // uphold about *resources*: when a request dies partway — cancelled,
    // disconnected, or failed inside the runner — every KV snapshot it touched
    // must end up either owned by a live cache entry or released. A snapshot
    // that is neither is a leaked multi-GB buffer, which is not a hypothetical
    // failure class here: an unreclaimed pool once drove decode from 35 to
    // 2.5 tok/s, recoverable only by restarting the process.
    //
    // Measuring live MLX memory needs a loaded model (tier 2). The *ownership*
    // half does not: the store already takes its runner as a trait, so a runner
    // that fails on its Nth call turns the whole cancellation sweep into a
    // tier-0 test, run at every interruption point rather than the two or three
    // anyone would write by hand.

    /// A runner that behaves normally until its `fail_at`-th call, then fails
    /// every call. Tracks snapshot ownership so a leak is detectable.
    struct InterruptingRunner {
        next_snap: u64,
        calls: usize,
        fail_at: usize,
        created: Vec<u64>,
        released: std::collections::HashSet<u64>,
    }

    impl InterruptingRunner {
        fn never_fails() -> Self {
            Self {
                next_snap: 0,
                calls: 0,
                fail_at: usize::MAX,
                created: Vec::new(),
                released: std::collections::HashSet::new(),
            }
        }

        fn failing_at(fail_at: usize) -> Self {
            Self {
                fail_at,
                ..Self::never_fails()
            }
        }

        /// Count this call and decide whether it is the one that dies.
        fn tick(&mut self) -> Result<()> {
            self.calls += 1;
            if self.calls >= self.fail_at {
                anyhow::bail!("interrupted at call {}", self.calls);
            }
            Ok(())
        }

        /// Snapshots handed out and not released — the resources still owned by
        /// someone.
        fn live(&self) -> Vec<u64> {
            self.created
                .iter()
                .copied()
                .filter(|id| !self.released.contains(id))
                .collect()
        }
    }

    impl SnapshotRunner for InterruptingRunner {
        fn prefill(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.tick()?;
            Ok((PREFILL_TOK, tokens.len()))
        }
        fn extend(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.tick()?;
            Ok((EXTEND_TOK, tokens.len()))
        }
        fn snapshot_state_deep(&mut self, _seq: u64) -> Result<(u64, usize)> {
            self.tick()?;
            let id = self.next_snap;
            self.next_snap += 1;
            self.created.push(id);
            Ok((id, 0))
        }
        fn fork_from_snapshot(&mut self, _snapshot_id: u64, _dst: u64) -> Result<usize> {
            self.tick()?;
            Ok(0)
        }
        fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
            // Releases are deliberately NOT interruptible: cleanup that can fail
            // would make every invariant below untestable, and the real
            // implementation's release is an in-process table removal.
            self.released.insert(snapshot_id);
            Ok(())
        }
    }

    /// The three ownership invariants, asserted together after any interruption.
    fn assert_snapshot_ownership_is_intact(
        store: &PrefixCacheStore,
        runner: &InterruptingRunner,
        ctx: &str,
    ) {
        let referenced: std::collections::HashSet<u64> = store
            .entries
            .values()
            .flatten()
            .map(|e| e.master_snapshot_id)
            .collect();

        for id in runner.live() {
            assert!(
                referenced.contains(&id),
                "{ctx}: snapshot {id} is neither released nor owned by a cache entry — \
                 a leaked KV buffer that only a restart reclaims"
            );
        }
        for e in store.entries.values().flatten() {
            assert!(
                !runner.released.contains(&e.master_snapshot_id),
                "{ctx}: entry points at snapshot {} which was already released — the \
                 next fork from this entry is a use-after-free",
                e.master_snapshot_id
            );
            assert_eq!(
                e.in_use, 0,
                "{ctx}: entry left in_use={} after the request died — the guard \
                 exempts it from eviction permanently, so this entry's KV can never \
                 be reclaimed",
                e.in_use
            );
        }
    }

    /// Drive `scenario` with a runner that dies at call 1, then 2, then 3, …
    /// until the whole scenario completes without being interrupted. Every
    /// interruption point must leave ownership intact and the store usable.
    fn sweep_interruptions(
        label: &str,
        scenario: fn(&mut PrefixCacheStore, &mut InterruptingRunner),
    ) {
        // Establish how many runner calls an uninterrupted run makes.
        let mut probe_store = PrefixCacheStore::with_policy(true, None, None);
        let mut probe = InterruptingRunner::never_fails();
        scenario(&mut probe_store, &mut probe);
        let total = probe.calls;
        assert!(total > 0, "{label}: scenario makes no runner calls");

        for fail_at in 1..=total {
            let mut store = PrefixCacheStore::with_policy(true, None, None);
            let mut runner = InterruptingRunner::failing_at(fail_at);
            // The scenario's own `unwrap`s are not used here: interruption is
            // expected, so the closure must tolerate errors (see the scenarios).
            scenario(&mut store, &mut runner);

            let ctx = format!("{label}: interrupted at runner call {fail_at}/{total}");
            assert_snapshot_ownership_is_intact(&store, &runner, &ctx);

            // Post-failure integrity: a fresh request on a healthy runner must
            // still work, which is the "next request succeeds normally" half of
            // the invariant.
            let mut healthy = InterruptingRunner::never_fails();
            let out = store.prefill_optionally_cached(
                &mut healthy,
                999,
                &[7, 7, 7],
                Some("recovery"),
                None,
            );
            assert!(
                out.is_ok(),
                "{ctx}: the store was unusable afterwards: {:?}",
                out.err()
            );
        }
    }

    #[test]
    fn cold_miss_survives_interruption_at_every_point() {
        sweep_interruptions("cold miss", |store, runner| {
            let _ = store.prefill_optionally_cached(runner, 1, &[1, 2, 3], Some("k"), None);
        });
    }

    #[test]
    fn extending_hit_survives_interruption_at_every_point() {
        sweep_interruptions("extending hit", |store, runner| {
            let _ = store.prefill_optionally_cached(runner, 1, &[1, 2, 3], Some("k"), None);
            let _ = store.prefill_optionally_cached(runner, 2, &[1, 2, 3, 4, 5], Some("k"), None);
        });
    }

    #[test]
    fn exact_retry_survives_interruption_at_every_point() {
        sweep_interruptions("exact retry", |store, runner| {
            let _ = store.prefill_optionally_cached(runner, 1, &[1, 2, 3], Some("k"), None);
            let _ = store.prefill_optionally_cached(runner, 2, &[1, 2, 3, 4], Some("k"), None);
            // Same prompt again — the exact-retry fork path.
            let _ = store.prefill_optionally_cached(runner, 3, &[1, 2, 3, 4], Some("k"), None);
        });
    }

    #[test]
    fn divergent_branches_survive_interruption_at_every_point() {
        sweep_interruptions("divergent branches", |store, runner| {
            let _ = store.prefill_optionally_cached(runner, 1, &[1, 2, 3], Some("k"), None);
            // Shares the head, then diverges — pushes a second branch.
            let _ = store.prefill_optionally_cached(runner, 2, &[1, 2, 9, 9], Some("k"), None);
            let _ = store.prefill_optionally_cached(runner, 3, &[1, 2, 3, 4], Some("k"), None);
        });
    }

    /// A cancelled request must not leave the entry it was forking from pinned.
    /// `in_use` is bumped across fork+extend precisely so a concurrent eviction
    /// cannot release the snapshot mid-flight; if an interruption skips the
    /// matching decrement, that entry becomes permanently unevictable and its
    /// KV is pinned for the life of the process.
    #[test]
    fn an_interrupted_fork_does_not_pin_its_source_entry_forever() {
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut warm = InterruptingRunner::never_fails();
        store
            .prefill_optionally_cached(&mut warm, 1, &[1, 2, 3], Some("k"), None)
            .expect("warm-up");
        assert_eq!(store.len(), 1);

        // Fail on the very first call of the second request: the fork itself.
        let mut runner = InterruptingRunner::failing_at(1);
        let res =
            store.prefill_optionally_cached(&mut runner, 2, &[1, 2, 3, 4, 5], Some("k"), None);
        assert!(res.is_err(), "the interrupted request reported success");

        let pinned: Vec<u32> = store
            .entries
            .values()
            .flatten()
            .map(|e| e.in_use)
            .filter(|&n| n > 0)
            .collect();
        assert!(
            pinned.is_empty(),
            "entries left pinned after an interrupted fork: {pinned:?} — eviction will \
             skip them forever"
        );

        // And eviction can in fact reclaim it now.
        let mut cleanup = InterruptingRunner::never_fails();
        assert_eq!(store.clear(&mut cleanup), 1);
    }

    #[test]
    fn no_key_is_plain_prefill_without_caching() {
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut m = MockRunner::default();
        let (tok, pos) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], None, None)
            .unwrap();
        assert_eq!((tok, pos), (PREFILL_TOK, 3));
        assert_eq!(m.log, vec!["prefill(3)"]);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn disabled_store_never_snapshots_even_with_key() {
        let mut store = PrefixCacheStore::with_policy(false, None, None);
        let mut m = MockRunner::default();
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap();
        assert_eq!(tok, PREFILL_TOK);
        assert_eq!(m.log, vec!["prefill(3)"]);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn miss_stores_snapshot_then_extend_hit_advances() {
        // Single-entry path must be bit-identical to the pre-multi-entry impl.
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut m = MockRunner::default();

        let (tok, pos) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap();
        assert_eq!((tok, pos), (PREFILL_TOK, 3));
        assert_eq!(m.log, vec!["prefill(3)", "snapshot->0"]);
        assert_eq!(store.len(), 1);

        // Extending HIT: prompt starts with cached [1,2,3] → fork, extend [4,5],
        // advance the (only) candidate in place, releasing snap 0.
        m.log.clear();
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 3, 4, 5], Some("k"), None)
            .unwrap();
        assert_eq!(tok, EXTEND_TOK);
        assert_eq!(
            m.log,
            vec!["fork(0)", "extend(2)", "snapshot->1", "release(0)"]
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn exact_retry_hit_forks_without_re_extending() {
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut m = MockRunner::default();

        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap();
        m.log.clear();

        let (tok, pos) = store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 3], Some("k"), None)
            .unwrap();
        assert_eq!((tok, pos), (PREFILL_TOK, 3));
        assert_eq!(m.log, vec!["fork(0)"]);
    }

    #[test]
    fn lru_cap_evicts_and_releases_oldest_snapshot() {
        let mut store = PrefixCacheStore::with_policy(true, None, Some(1));
        let mut m = MockRunner::default();

        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("a"), None)
            .unwrap();
        m.log.clear();
        store
            .prefill_optionally_cached(&mut m, 2, &[9, 9, 9], Some("b"), None)
            .unwrap();
        assert_eq!(store.len(), 1);
        assert!(
            m.log.contains(&"release(0)".to_string()),
            "evicted entry's snapshot must be released, got {:?}",
            m.log
        );
    }

    // ── Multi-entry LPM tests ──────────────────────────────────────────────

    #[test]
    fn lpm_selects_longer_matching_branch() {
        // Two candidates under one key sharing a head [1,2] then diverging:
        // branch A = [1,2,3], branch B = [1,2,7,8]. A prompt [1,2,7,8,9] must
        // fork B (the longer match), not A.
        let mut store = PrefixCacheStore::with_branches(true, None, None, 4);
        let mut m = MockRunner::default();

        // Seed branch A via cold MISS.
        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap(); // snapshot->0
        // Seed branch B: diverges from A in the tail → NEW branch, no supersede.
        store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 7, 8], Some("k"), None)
            .unwrap(); // snapshot->1
        assert_eq!(store.len(), 2, "divergent prompt must create a 2nd branch");

        m.log.clear();
        // Prompt extends B → fork snap 1 (B), not snap 0 (A).
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 3, &[1, 2, 7, 8, 9], Some("k"), None)
            .unwrap();
        assert_eq!(tok, EXTEND_TOK);
        assert!(
            m.log.iter().any(|l| l == "fork(1)"),
            "must fork the longer-matching branch B (snap 1), got {:?}",
            m.log
        );
        assert!(
            !m.log.iter().any(|l| l == "fork(0)"),
            "must not fork branch A, got {:?}",
            m.log
        );
    }

    #[test]
    fn divergent_prompt_branches_instead_of_overwriting() {
        let mut store = PrefixCacheStore::with_branches(true, None, None, 4);
        let mut m = MockRunner::default();

        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap();
        assert_eq!(store.len(), 1);
        // Shares head [1,2] but diverges (3 vs 9) → branch, keep both.
        store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 9], Some("k"), None)
            .unwrap();
        assert_eq!(store.len(), 2);

        // Both lines are still independently reusable.
        m.log.clear();
        let (a, _) = store
            .prefill_optionally_cached(&mut m, 3, &[1, 2, 3, 4], Some("k"), None)
            .unwrap();
        let (b, _) = store
            .prefill_optionally_cached(&mut m, 4, &[1, 2, 9, 4], Some("k"), None)
            .unwrap();
        assert_eq!((a, b), (EXTEND_TOK, EXTEND_TOK));
    }

    #[test]
    fn branch_cap_evicts_least_recent_branch() {
        // cap branches=2: a 3rd divergent branch evicts the LRU branch.
        let mut store = PrefixCacheStore::with_branches(true, None, None, 2);
        let mut m = MockRunner::default();

        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap(); // snap 0 (branch A, oldest)
        store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 9], Some("k"), None)
            .unwrap(); // snap 1 (branch B)
        assert_eq!(store.len(), 2);
        m.log.clear();
        store
            .prefill_optionally_cached(&mut m, 3, &[1, 2, 5], Some("k"), None)
            .unwrap(); // snap 2 (branch C) → cap=2 evicts A
        assert_eq!(store.len(), 2, "branch cap must hold len at 2");
        assert!(
            m.log.contains(&"release(0)".to_string()),
            "least-recent branch A (snap 0) must be released, got {:?}",
            m.log
        );
    }

    #[test]
    fn supersede_keeps_single_entry_on_same_line() {
        // A MISS whose prompt extends an existing candidate on the SAME line
        // supersedes it in place (no new branch) — preserving single-entry
        // identity. (Reached when the HIT path is skipped, e.g. last_token-less
        // boundary entries via store_master, exercised here through the tools
        // MISS using a non-extending relationship is not it; we test the helper
        // path via store_master below.)
        let mut store = PrefixCacheStore::with_branches(true, None, None, 4);
        let mut m = MockRunner::default();
        // boundary-style stores (last_token=None) on the same growing line.
        store.store_master(&mut m, "k", 10, vec![1, 2], None);
        assert_eq!(store.len(), 1);
        store.store_master(&mut m, "k", 11, vec![1, 2, 3], None);
        // [1,2,3] starts_with [1,2] → supersede in place, release snap 10.
        assert_eq!(store.len(), 1);
        assert!(m.log.contains(&"release(10)".to_string()));
    }

    #[test]
    fn get_master_for_returns_longest_prefix() {
        let mut store = PrefixCacheStore::with_branches(true, None, None, 4);
        let mut m = MockRunner::default();
        // Two genuinely divergent branches sharing head [1,2]: A=[1,2,3,4],
        // B=[1,2,7,8]. Neither is a prefix of the other, so both are kept.
        store.store_master(&mut m, "k", 10, vec![1, 2, 3, 4], None);
        store.store_master(&mut m, "k", 11, vec![1, 2, 7, 8], None);
        assert_eq!(store.len(), 2);
        // prompt extending B → longest matching prefix is B (snap 11).
        let got = store.get_master_for("k", &[1, 2, 7, 8, 9]);
        assert_eq!(got, Some((11, vec![1, 2, 7, 8])));
        // prompt extending A → longest matching prefix is A (snap 10).
        let got2 = store.get_master_for("k", &[1, 2, 3, 4, 5]);
        assert_eq!(got2, Some((10, vec![1, 2, 3, 4])));
    }

    // ── Phase 0: incremental chunked-prefill boundary caching ──────────────

    #[test]
    fn incremental_off_is_byte_identical_miss() {
        // Feature OFF: a MISS with a boundary supplied must be the exact original
        // single-prefill path — no boundary snapshot, one branch.
        let mut store = PrefixCacheStore::with_incremental(true, 4, false);
        let mut m = MockRunner::default();
        let (tok, pos) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();
        assert_eq!((tok, pos), (PREFILL_TOK, 4));
        assert_eq!(m.log, vec!["prefill(4)", "snapshot->0"]);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn incremental_on_snapshots_boundary_branch() {
        // Feature ON: cold-prefill the head [1,2], snapshot it (pinned), extend
        // the tail [3,4], snapshot the full prompt. Two branches.
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let mut m = MockRunner::default();
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();
        // The full prompt's returned token comes from the tail extend.
        assert_eq!(tok, EXTEND_TOK);
        assert_eq!(
            m.log,
            vec!["prefill(2)", "snapshot->0", "extend(2)", "snapshot->1"]
        );
        assert_eq!(store.len(), 2, "head boundary + full prompt branches");
    }

    #[test]
    fn incremental_boundary_lets_divergent_prompt_hit() {
        // The headline win: after a [1,2,3,4] MISS that snapshotted the [1,2]
        // boundary, a divergent [1,2,9,9] sharing only the head must HIT the
        // boundary (fork snap 0 + extend the [9,9] tail), NOT cold-prefill.
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let mut m = MockRunner::default();
        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();

        m.log.clear();
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 9, 9], Some("k"), Some(2))
            .unwrap();
        assert_eq!(tok, EXTEND_TOK);
        assert!(
            m.log.iter().any(|l| l == "fork(0)"),
            "must fork the [1,2] boundary snapshot, got {:?}",
            m.log
        );
        assert!(
            m.log.iter().any(|l| l == "extend(2)"),
            "must extend only the divergent [9,9] tail, got {:?}",
            m.log
        );
        assert!(
            !m.log.iter().any(|l| l.starts_with("prefill(")),
            "must NOT cold-prefill, got {:?}",
            m.log
        );
        // The [1,2] boundary survives the divergent advance so a THIRD divergent
        // prompt can still reuse it (head + 2 full-prompt tails).
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn incremental_full_prompt_branch_preferred_over_boundary() {
        // A prompt extending the FULL branch must fork the full snapshot (snap 1),
        // not the shorter [1,2] boundary (snap 0) — LPM picks the longest match.
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let mut m = MockRunner::default();
        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();
        m.log.clear();
        store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 3, 4, 5], Some("k"), Some(2))
            .unwrap();
        assert!(
            m.log.iter().any(|l| l == "fork(1)"),
            "must fork the longer full-prompt branch, got {:?}",
            m.log
        );
        assert!(
            !m.log.iter().any(|l| l == "fork(0)"),
            "must not fork the shorter boundary, got {:?}",
            m.log
        );
    }

    #[test]
    fn pinned_boundary_survives_full_prompt_supersede() {
        // Inserting the full-prompt branch must NOT release the [1,2] boundary,
        // even though [1,2,3,4] starts_with [1,2] (which would normally supersede).
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let mut m = MockRunner::default();
        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();
        assert!(
            !m.log.iter().any(|l| l == "release(0)"),
            "the pinned boundary snapshot 0 must not be released, got {:?}",
            m.log
        );
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn incremental_respects_branch_cap() {
        // cap=2: MISS creates [1,2]+[1,2,3,4] (at cap). A divergent HIT adds a
        // 3rd branch, evicting the LRU non-pinned (the full-prompt branch),
        // while the just-accessed boundary survives.
        let mut store = PrefixCacheStore::with_incremental(true, 2, true);
        let mut m = MockRunner::default();
        store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4], Some("k"), Some(2))
            .unwrap();
        assert_eq!(store.len(), 2);
        m.log.clear();
        store
            .prefill_optionally_cached(&mut m, 2, &[1, 2, 9, 9], Some("k"), Some(2))
            .unwrap();
        assert_eq!(store.len(), 2, "branch cap holds len at 2");
        assert!(
            m.log.contains(&"release(1)".to_string()),
            "LRU full-prompt branch (snap 1) must be evicted, got {:?}",
            m.log
        );
    }

    // ── L2 disk tier (persist / rehydrate) wiring ──────────────────────────

    /// Mock that logs the disk-tier hooks and can return one canned rehydrated
    /// snapshot, so tests pin down the persist-on-MISS and load-on-MISS control
    /// flow without a real runner. Kept separate from `MockRunner` so the
    /// existing (disk-off, default no-op) tests stay byte-identical.
    #[derive(Default)]
    struct PersistMockRunner {
        log: Vec<String>,
        next_snap: u64,
        /// Returned (once) by `load_persisted_snapshot` to simulate a disk HIT.
        rehydrate: Option<(u64, Vec<u32>, Option<u32>)>,
    }

    impl SnapshotRunner for PersistMockRunner {
        fn prefill(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.log.push(format!("prefill({})", tokens.len()));
            Ok((PREFILL_TOK, tokens.len()))
        }
        fn extend(&mut self, _seq: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.log.push(format!("extend({})", tokens.len()));
            Ok((EXTEND_TOK, tokens.len()))
        }
        fn snapshot_state_deep(&mut self, _seq: u64) -> Result<(u64, usize)> {
            let id = self.next_snap;
            self.next_snap += 1;
            self.log.push(format!("snapshot->{id}"));
            Ok((id, 0))
        }
        fn fork_from_snapshot(&mut self, snapshot_id: u64, _dst: u64) -> Result<usize> {
            self.log.push(format!("fork({snapshot_id})"));
            Ok(0)
        }
        fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
            self.log.push(format!("release({snapshot_id})"));
            Ok(())
        }
        fn persist_snapshot(
            &mut self,
            snapshot_id: u64,
            _key: &str,
            prefix_tokens: &[u32],
            last_token: Option<u32>,
        ) -> Result<()> {
            self.log.push(format!(
                "persist({snapshot_id},{},{last_token:?})",
                prefix_tokens.len()
            ));
            Ok(())
        }
        fn load_persisted_snapshot(
            &mut self,
            _key: &str,
            _prompt_ids: &[u32],
        ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
            match self.rehydrate.take() {
                Some(r) => {
                    self.log.push("load->hit".to_string());
                    Ok(Some(r))
                }
                None => {
                    self.log.push("load->miss".to_string());
                    Ok(None)
                }
            }
        }
    }

    #[test]
    fn cold_miss_persists_snapshot_to_disk_tier() {
        // On an in-memory miss the driver first probes the disk tier (miss),
        // cold-prefills, snapshots, then persists the fresh snapshot durably.
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut m = PersistMockRunner::default();
        let (tok, pos) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3], Some("k"), None)
            .unwrap();
        assert_eq!((tok, pos), (PREFILL_TOK, 3));
        assert_eq!(
            m.log,
            vec![
                "load->miss".to_string(),
                "prefill(3)".to_string(),
                "snapshot->0".to_string(),
                "persist(0,3,Some(100))".to_string(),
            ]
        );
    }

    #[test]
    fn disk_rehydrate_avoids_cold_prefill_on_restart() {
        // Simulates a fresh process: the in-memory store is empty, but the disk
        // tier has a snapshot for the [1,2,3] prefix. A [1,2,3,4,5] prompt must
        // rehydrate (load->hit), fork the disk snapshot, and extend only the
        // [4,5] suffix — never cold-prefilling.
        let mut store = PrefixCacheStore::with_policy(true, None, None);
        let mut m = PersistMockRunner {
            rehydrate: Some((42, vec![1, 2, 3], Some(100))),
            ..Default::default()
        };
        let (tok, _) = store
            .prefill_optionally_cached(&mut m, 1, &[1, 2, 3, 4, 5], Some("k"), None)
            .unwrap();
        assert_eq!(tok, EXTEND_TOK);
        assert!(m.log.contains(&"load->hit".to_string()), "log={:?}", m.log);
        assert!(m.log.contains(&"fork(42)".to_string()), "log={:?}", m.log);
        assert!(m.log.contains(&"extend(2)".to_string()), "log={:?}", m.log);
        assert!(
            !m.log.iter().any(|l| l.starts_with("prefill(")),
            "must not cold-prefill after disk rehydrate, log={:?}",
            m.log
        );
    }

    fn fake_prefix_entry(snapshot_id: u64, last_access: Instant) -> PrefixCacheEntry {
        PrefixCacheEntry {
            master_snapshot_id: snapshot_id,
            prefix_tokens: vec![1, 2, 3],
            last_access,
            hits: 0,
            last_token: None,
            in_use: 0,
            pinned_boundary: false,
        }
    }

    #[test]
    fn pick_prefix_eviction_no_limits_keeps_all() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert("a".to_string(), vec![fake_prefix_entry(11, now)]);
        map.insert("b".to_string(), vec![fake_prefix_entry(12, now)]);
        let victims = pick_eviction_victims(&map, now, None, None);
        assert!(victims.is_empty());
    }

    #[test]
    fn pick_prefix_eviction_lru_drops_oldest_until_cap() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert(
            "old".to_string(),
            vec![fake_prefix_entry(1, now - Duration::from_secs(30))],
        );
        map.insert("new".to_string(), vec![fake_prefix_entry(2, now)]);
        let victims = pick_eviction_victims(&map, now, None, Some(1));
        let keys: Vec<&str> = victims.iter().map(|(k, _, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["old"]);
        assert!(victims[0].2.contains("LRU"));
    }

    #[test]
    fn pick_prefix_eviction_ttl_drops_stale_only() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert("fresh".to_string(), vec![fake_prefix_entry(1, now)]);
        map.insert(
            "stale".to_string(),
            vec![fake_prefix_entry(2, now - Duration::from_secs(120))],
        );
        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(60)), None);
        let keys: Vec<&str> = victims.iter().map(|(k, _, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["stale"]);
        assert!(victims[0].2.contains("TTL"));
    }

    #[test]
    fn pick_prefix_eviction_never_evicts_in_use() {
        let mut map = HashMap::new();
        let now = Instant::now();
        let mut locked = fake_prefix_entry(1, now - Duration::from_secs(300));
        locked.in_use = 1;
        map.insert("locked".to_string(), vec![locked]);
        // TTL would normally evict it (stale), but it is in use.
        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(60)), Some(1));
        assert!(
            victims.is_empty(),
            "in-use candidate must never be evicted, got {:?}",
            victims
        );
    }
}
