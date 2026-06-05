//! Track A1 prefix-cache machinery — extracted backend-agnostic.
//!
//! A [`PrefixCacheStore`] holds master KV snapshots taken at the end of a
//! *shared* prompt prefix (typically the system-prompt block), keyed by a
//! process-stable hash. [`PrefixCacheStore::prefill_optionally_cached`] is the
//! HIT / MISS / exact-retry driver: on HIT it forks the stored snapshot and
//! extends only the suffix; on MISS it cold-prefills and snapshots the result
//! so the next request can fork from it.
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
}

/// Keyed store of prefix-cache master snapshots plus its TTL / LRU policy.
pub struct PrefixCacheStore {
    entries: HashMap<String, PrefixCacheEntry>,
    enabled: bool,
    ttl: Option<Duration>,
    max: Option<usize>,
}

impl PrefixCacheStore {
    /// Construct from the `LUMEN_MLX_PREFIX_CACHE*` env trio.
    pub fn from_env() -> Self {
        let (enabled, ttl, max) = read_limits();
        Self {
            entries: HashMap::new(),
            enabled,
            ttl,
            max,
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
    /// prefix), forks the stored snapshot and extends by the suffix, skipping
    /// re-processing the common prefix. On MISS, runs a normal prefill and
    /// snapshots the result under `key` so the next request can fork from it.
    ///
    /// Falls through to a plain `runner.prefill` when `key` is `None` or the
    /// feature is disabled. Snapshot failures are logged but non-fatal — the
    /// request completes, just without (re)populating the cache.
    pub fn prefill_optionally_cached<R: SnapshotRunner>(
        &mut self,
        runner: &mut R,
        seq_id: u64,
        prompt_ids: &[u32],
        key: Option<&str>,
    ) -> Result<(u32, usize)> {
        let key = match key.filter(|_| self.enabled) {
            Some(k) => k,
            None => return runner.prefill(seq_id, prompt_ids),
        };
        self.evict_stale(runner);

        let cached = self
            .entries
            .get(key)
            .map(|e| (e.master_snapshot_id, e.prefix_tokens.clone(), e.last_token));

        // HIT when the new prompt *extends* the cached prefix, OR is byte-for-byte
        // identical to it (empty suffix) AND we captured that prefix's argmax
        // token. The identical case is the retry loop: a client whose
        // time-to-first-token timeout (omp default 100s) fires mid cold-prefill
        // drops the socket and resends the SAME request. Without the `==` arm it
        // MISSes and re-runs the full ~145s prefill on every retry — an infinite
        // loop. With the stored argmax we fork the completed snapshot and start
        // decoding immediately.
        if let Some((master, prefix, cached_last)) = cached.filter(|(_, p, last)| {
            !p.is_empty()
                && prompt_ids.starts_with(p)
                && (prompt_ids.len() > p.len() || last.is_some())
        }) {
            let suffix = &prompt_ids[prefix.len()..];
            let t = Instant::now();
            let _ = runner.fork_from_snapshot(master, seq_id)?;
            let (last, pos) = if suffix.is_empty() {
                // Exact-prompt retry: the snapshot's KV already covers the whole
                // prompt, so there is nothing to extend. `cached_last` is `Some`
                // here (guaranteed by the filter), and equals what a cold prefill
                // of this prompt would have returned (argmax of the final logits).
                (
                    cached_last.expect("filter guarantees Some on empty suffix"),
                    prefix.len(),
                )
            } else {
                runner.extend(seq_id, suffix)?
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            // ADVANCE the snapshot to THIS turn's full prompt so the NEXT turn
            // forks from here and re-prefills only its own new tokens. Without
            // this the snapshot stays frozen at the first turn's prompt and the
            // reused `prefix` never grows — so every turn re-prefills the whole
            // (growing) conversation suffix (observed: suffix 39→1265→9327→…).
            // The trailing generation header is a clean prefix of the next turn
            // because a completed assistant turn re-renders with the same
            // `<think>\n\n</think>\n\n` block, so storing the full prompt is safe.
            // Skipped on an empty suffix: the forked state already equals the
            // stored snapshot, so re-snapshotting would only churn ~3GB of KV.
            let prev_hits = self.entries.get(key).map(|e| e.hits).unwrap_or(0);
            let mut advanced_to = prefix.len();
            if suffix.is_empty() {
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.last_access = Instant::now();
                    entry.hits += 1;
                }
            } else {
                match runner.snapshot_state_deep(seq_id) {
                    Ok((snap_id, _)) => {
                        if let Some(old) = self.entries.insert(
                            key.to_string(),
                            PrefixCacheEntry {
                                master_snapshot_id: snap_id,
                                prefix_tokens: prompt_ids.to_vec(),
                                last_access: Instant::now(),
                                hits: prev_hits + 1,
                                last_token: Some(last),
                            },
                        ) {
                            let _ = runner.release_snapshot(old.master_snapshot_id);
                        }
                        self.evict_stale(runner);
                        advanced_to = prompt_ids.len();
                    }
                    Err(e) => {
                        // Snapshot failed — keep the old (frozen) entry, just bump.
                        if let Some(entry) = self.entries.get_mut(key) {
                            entry.last_access = Instant::now();
                            entry.hits += 1;
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

        // Cold prefill, then snapshot under `key` so the next request with
        // the same key + extended prompt can fork from here.
        let (last, pos) = runner.prefill(seq_id, prompt_ids)?;
        match runner.snapshot_state_deep(seq_id) {
            Ok((snap_id, _snap_pos)) => {
                if let Some(old) = self.entries.remove(key) {
                    let _ = runner.release_snapshot(old.master_snapshot_id);
                }
                self.entries.insert(
                    key.to_string(),
                    PrefixCacheEntry {
                        master_snapshot_id: snap_id,
                        prefix_tokens: prompt_ids.to_vec(),
                        last_access: Instant::now(),
                        hits: 0,
                        last_token: Some(last),
                    },
                );
                self.evict_stale(runner);
                eprintln!(
                    "[mlx] prefix-cache MISS (tools) key={key:?} stored snapshot={snap_id} prefix_len={}",
                    prompt_ids.len()
                );
            }
            Err(e) => {
                eprintln!("[mlx] prefix-cache snapshot skipped (tools, key={key:?}): {e:#}");
            }
        }
        Ok((last, pos))
    }

    /// Drop entries exceeding the TTL / LRU policy, releasing their snapshots.
    /// No-op when both TTL and max are unset.
    pub fn evict_stale<R: SnapshotRunner>(&mut self, runner: &mut R) {
        let now = Instant::now();
        let victims = pick_eviction_victims(&self.entries, now, self.ttl, self.max);
        for (key, reason) in victims {
            if let Some(entry) = self.entries.remove(&key) {
                let _ = runner.release_snapshot(entry.master_snapshot_id);
                eprintln!("[mlx] prefix-cache key={key:?} evicted ({reason})");
            }
        }
    }

    /// Look up a master snapshot for `key`, returning `(snapshot_id, prefix
    /// tokens)`. Used by the streaming boundary path to decide fork-vs-cold.
    pub fn get_master(&self, key: &str) -> Option<(u64, Vec<u32>)> {
        self.entries
            .get(key)
            .map(|e| (e.master_snapshot_id, e.prefix_tokens.clone()))
    }

    /// Replace the entry under `key` with a fresh master snapshot, releasing any
    /// prior snapshot, then enforce TTL / LRU. `hits` resets to 0. Used by the
    /// streaming boundary path (which passes `last_token = None`).
    pub fn store_master<R: SnapshotRunner>(
        &mut self,
        runner: &mut R,
        key: &str,
        snapshot_id: u64,
        prefix_tokens: Vec<u32>,
        last_token: Option<u32>,
    ) {
        if let Some(old) = self.entries.remove(key) {
            let _ = runner.release_snapshot(old.master_snapshot_id);
        }
        self.entries.insert(
            key.to_string(),
            PrefixCacheEntry {
                master_snapshot_id: snapshot_id,
                prefix_tokens,
                last_access: Instant::now(),
                hits: 0,
                last_token,
            },
        );
        self.evict_stale(runner);
    }

    /// Drop one entry, releasing its snapshot. Returns its `hits` count if the
    /// entry existed.
    pub fn drop_entry<R: SnapshotRunner>(&mut self, runner: &mut R, key: &str) -> Option<u64> {
        let entry = self.entries.remove(key)?;
        let hits = entry.hits;
        let _ = runner.release_snapshot(entry.master_snapshot_id);
        Some(hits)
    }

    /// Drop all entries, releasing every snapshot. Returns the number released.
    pub fn clear<R: SnapshotRunner>(&mut self, runner: &mut R) -> usize {
        let n = self.entries.len();
        let entries: Vec<_> = self.entries.drain().collect();
        for (_, entry) in entries {
            let _ = runner.release_snapshot(entry.master_snapshot_id);
        }
        n
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Read A1 prefix-cache config from env.
///
/// - `LUMEN_MLX_PREFIX_CACHE=0`          — opt OUT (default ON since v0.4.7).
/// - `LUMEN_MLX_PREFIX_CACHE_TTL_SECS=N` — drop entries idle > N seconds.
/// - `LUMEN_MLX_PREFIX_CACHE_MAX=N`      — keep at most N entries, LRU-evict.
///
/// Default is ON: the feature has been validated since 2026-05-18 and provides
/// the 5-6× speedup users expect for repeated chat turns with a shared system
/// prompt. The env var is the escape hatch for debugging, not the gate.
fn read_limits() -> (bool, Option<Duration>, Option<usize>) {
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
    (enabled, ttl, max)
}

/// Pure TTL-then-LRU victim selection. Returns `(key, reason)` pairs to evict.
/// Kept side-effect-free for unit testing; the caller releases snapshots.
fn pick_eviction_victims(
    entries: &HashMap<String, PrefixCacheEntry>,
    now: Instant,
    ttl: Option<Duration>,
    max: Option<usize>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut alive: HashMap<String, Instant> = entries
        .iter()
        .map(|(k, e)| (k.clone(), e.last_access))
        .collect();

    if let Some(ttl) = ttl {
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

    if let Some(max) = max {
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

    fn fake_prefix_entry(snapshot_id: u64, last_access: Instant) -> PrefixCacheEntry {
        PrefixCacheEntry {
            master_snapshot_id: snapshot_id,
            prefix_tokens: vec![1, 2, 3],
            last_access,
            hits: 0,
            last_token: None,
        }
    }

    #[test]
    fn pick_prefix_eviction_no_limits_keeps_all() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert("a".to_string(), fake_prefix_entry(11, now));
        map.insert("b".to_string(), fake_prefix_entry(12, now));
        let victims = pick_eviction_victims(&map, now, None, None);
        assert!(victims.is_empty());
    }

    #[test]
    fn pick_prefix_eviction_lru_drops_oldest_until_cap() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert(
            "old".to_string(),
            fake_prefix_entry(1, now - Duration::from_secs(30)),
        );
        map.insert("new".to_string(), fake_prefix_entry(2, now));
        let victims = pick_eviction_victims(&map, now, None, Some(1));
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["old"]);
        assert!(victims[0].1.contains("LRU"));
    }

    #[test]
    fn pick_prefix_eviction_ttl_drops_stale_only() {
        let mut map = HashMap::new();
        let now = Instant::now();
        map.insert("fresh".to_string(), fake_prefix_entry(1, now));
        map.insert(
            "stale".to_string(),
            fake_prefix_entry(2, now - Duration::from_secs(120)),
        );
        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(60)), None);
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["stale"]);
        assert!(victims[0].1.contains("TTL"));
    }
}
