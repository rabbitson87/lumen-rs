//! Qwen3.8-Flash-Next as one of the Qwen backend's runners: sequences, prefill
//! in chunks, greedy and sampled decode, snapshots and rollback points.
//!
//! Everything the chat layer does with a runner (templates, tool calls, sampling,
//! sessions, the prefix cache) is the Qwen family's own and comes with the
//! `Runner` impl; this file only has to say how a sequence advances and how a
//! state is copied. A sequence's state is a [`Qwen4ExpCache`], a bag of
//! immutable MLX array handles plus the KV buffers, so a snapshot or a rollback
//! point is a clone that shares storage until either side appends.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Array;

use super::model::{LoadOptions, NgramSource, Qwen4ExpCache, Qwen4ExpModel};
use crate::kv_disk::{DiskKvStore, KvManifest};
use crate::session_feed::PrefillGrid;
use crate::{ProbeRows, Runner};

/// What the backend needs to know after a load.
pub(crate) struct LoadInfo {
    pub eos_tokens: Vec<u32>,
    pub vocab_size: usize,
}

struct Seq {
    cache: Qwen4ExpCache,
    position: usize,
    rollback: Option<(Qwen4ExpCache, usize)>,
    /// The disk entry this sequence last wrote or was restored from, so a longer
    /// state replaces it instead of piling up beside it.
    persisted_key: Option<String>,
}

pub(crate) struct Qwen4ExpRunner {
    model: Qwen4ExpModel,
    seqs: HashMap<u64, Seq>,
    snapshots: HashMap<u64, (Qwen4ExpCache, usize)>,
    next_snapshot: u64,
    chunk: usize,
    /// L2 tier (`LUMEN_KV_DISK`): session states at prefill-grid points, so a
    /// restarted server resumes a conversation without reading the model again.
    disk: Option<DiskKvStore>,
    fingerprint: String,
}

/// Tokens per prefill chunk. The chunks of a prompt go through the model layer by layer, so
/// the SSD reads each layer's experts once for the whole prompt whatever the chunk size
/// (measured on an 18K-token prompt, 6.7 GB pool: 234 / 232-241 / 227 tok/s at chunks of 512 /
/// 2,048 / 4,096, where feeding chunk by chunk gave 73 / 153 / 145). The chunk only sets the
/// activation memory (and how close to a conversation's end a persisted state can sit), so it
/// is small: a chunk's memory is what the pool gives up.
const PREFILL_CHUNK: usize = 1024;

/// What the resident trunk (embeddings, attention, DeltaNet, hyper-connections,
/// routers, shared experts) occupies.
const TRUNK_BYTES: u64 = 3_600_000_000;

/// The offsets, within `len` tokens that start at absolute position `start`, where chunks of
/// `chunk` end when chunk boundaries sit on multiples of `chunk` counted from position 0 (the
/// grid a cold prefill uses). The last cut is `len`.
fn chunk_cuts(start: usize, len: usize, chunk: usize) -> Vec<usize> {
    let chunk = chunk.max(1);
    let mut cuts: Vec<usize> = ((start / chunk + 1)..)
        .map(|k| k * chunk - start)
        .take_while(|&c| c < len)
        .collect();
    cuts.push(len);
    cuts
}

/// Memory a prefill chunk needs beyond the trunk and the expert pool: ~1.1 MB
/// per token (measured: a 2,460-token chunk peaked 2.5 GB above trunk + pool, a
/// 1,024-token one 1.3 GB) plus slack for MLX's buffer cache.
fn prefill_reserve(chunk: usize) -> u64 {
    2_000_000_000 + chunk as u64 * 1_100_000
}

/// The chunk and the expert pool for a Mac whose GPU may use `usable` bytes: what the working
/// set leaves after the trunk and one chunk's activations.
fn plan_memory(usable: u64, chunk_override: Option<usize>) -> (usize, u64) {
    let chunk = chunk_override.unwrap_or(PREFILL_CHUNK);
    let pool = usable.saturating_sub(TRUNK_BYTES + prefill_reserve(chunk));
    (chunk, pool)
}

fn chunk_override() -> Option<usize> {
    std::env::var("LUMEN_QWEN35_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&c| c > 0)
}

/// Bytes of expert pool: `LUMEN_MOE_EXPERT_CACHE_GB` (a number of GB, or `auto`),
/// and when unset what the GPU working set leaves after the trunk and a prefill
/// chunk's activations. Holding all experts in memory is not an option on a Mac
/// (they are 68 GB).
fn expert_budget_bytes(usable: u64, chunk: usize, auto: u64) -> Result<u64> {
    if let Ok(raw) = std::env::var("LUMEN_MOE_EXPERT_CACHE_GB") {
        let raw = raw.trim();
        if !raw.is_empty() && !raw.eq_ignore_ascii_case("auto") {
            let gb: f64 = raw
                .parse()
                .ok()
                .filter(|g: &f64| g.is_finite() && *g > 0.0)
                .ok_or_else(|| {
                    anyhow!(
                        "LUMEN_MOE_EXPERT_CACHE_GB={raw:?} is not a positive number of GB or `auto`"
                    )
                })?;
            let bytes = (gb * 1e9) as u64;
            if bytes + TRUNK_BYTES + prefill_reserve(chunk) > usable {
                eprintln!(
                    "[qwen4_exp] WARNING: a {gb} GB pool plus the trunk and a {chunk}-token prefill \
                     chunk needs {:.1} GB; this GPU allows {:.1} GB, so a long prompt may fail with \
                     `Insufficient Memory`. Lower the pool or LUMEN_QWEN35_PREFILL_CHUNK.",
                    (bytes + TRUNK_BYTES + prefill_reserve(chunk)) as f64 / 1e9,
                    usable as f64 / 1e9
                );
            }
            return Ok(bytes);
        }
    }
    Ok(auto)
}

impl Qwen4ExpRunner {
    pub(crate) fn load(dir: &Path) -> Result<(Self, LoadInfo)> {
        let ws = crate::metal_memory::working_set_limit().unwrap_or(16_000_000_000) as f64;
        let usable = (ws * crate::metal_memory::WORKING_SET_SHARE) as u64;
        let (chunk, auto_pool) = plan_memory(usable, chunk_override());
        let budget = expert_budget_bytes(usable, chunk, auto_pool)?;
        eprintln!(
            "[qwen4_exp] GPU may use {:.1} GB: expert pool {:.2} GB, prefill chunk {chunk} tokens \
             (LUMEN_MOE_EXPERT_CACHE_GB / LUMEN_QWEN35_PREFILL_CHUNK override)",
            usable as f64 / 1e9,
            budget as f64 / 1e9
        );
        let model = Qwen4ExpModel::load_with(
            dir,
            LoadOptions {
                expert_budget_bytes: Some(budget),
                ngram: NgramSource::Auto,
            },
        )
        .with_context(|| format!("qwen4_exp: load `{}`", dir.display()))?;
        let info = LoadInfo {
            eos_tokens: model.config().eos_tokens.clone(),
            vocab_size: model.config().vocab_size,
        };
        // A persisted state is only valid for the same weights, MLX build and prefill grid.
        let mut settings = crate::kv_disk::identity_settings_from_env();
        settings.push(("mlx".into(), crate::kv_disk::mlx_version()));
        settings.push(("qwen4_exp_prefill_chunk".into(), chunk.to_string()));
        let fingerprint = crate::kv_disk::kv_identity(&dir.display().to_string(), dir, &settings);
        let disk = crate::kv_disk::open_store_from_env(&fingerprint, "qwen4_exp");
        Ok((
            Self {
                model,
                seqs: HashMap::new(),
                snapshots: HashMap::new(),
                next_snapshot: 1,
                chunk,
                disk,
                fingerprint,
            },
            info,
        ))
    }

    /// Run `tokens` through `cache` in chunks of `self.chunk` (counted from the first token) and
    /// return the last position's logits `[1, 1, vocab]`. The chunks go through the model layer by
    /// layer, so each layer's experts are read once for the whole prompt rather than once per chunk.
    fn forward_chunked(&self, tokens: &[u32], cache: &mut Qwen4ExpCache) -> Result<Array> {
        if tokens.is_empty() {
            bail!("qwen4_exp: no tokens to feed");
        }
        let cuts = chunk_cuts(0, tokens.len(), self.chunk);
        let (logits, _) = self.model.forward_pieces(tokens, cache, &cuts, None)?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        Ok(logits)
    }

    /// Feed `tokens` (the start of a sequence when `fresh`, else its continuation) in chunks
    /// that end on the absolute prefill grid, leaving the rollback point `mark` tokens in. That
    /// is what the session's cut-by-cut feed produces, in one layer-major pass.
    pub(crate) fn feed_with_mark(
        &mut self,
        seq_id: u64,
        fresh: bool,
        tokens: &[u32],
        mark: usize,
    ) -> Result<Option<(u32, usize)>> {
        let start = if fresh {
            0
        } else {
            self.seq(seq_id, "feed")?.position
        };
        let cuts = chunk_cuts(start, tokens.len(), self.chunk);
        if tokens.is_empty() || !cuts.contains(&mark) {
            return Ok(None);
        }
        let mut cache = if fresh {
            self.model.make_cache()
        } else {
            self.seq(seq_id, "feed")?.cache.clone()
        };
        let (logits, point) = self
            .model
            .forward_pieces(tokens, &mut cache, &cuts, Some(mark))?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        let next = Self::argmax(&logits)?;
        let position = start + tokens.len();
        let point = point.map(|p| (p, start + mark));
        match self.seqs.get_mut(&seq_id) {
            Some(seq) if !fresh => {
                seq.cache = cache;
                seq.position = position;
                seq.rollback = point;
            }
            _ => {
                self.seqs.insert(
                    seq_id,
                    Seq {
                        cache,
                        position,
                        rollback: point,
                        persisted_key: None,
                    },
                );
            }
        }
        Ok(Some((next, position)))
    }

    fn argmax(logits: &Array) -> Result<u32> {
        let flat = mlx_rs::ops::reshape(logits, &[-1])?;
        let idx = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false)?;
        idx.eval().context("qwen4_exp: argmax")?;
        Ok(idx.item::<u32>())
    }

    fn seq(&mut self, seq_id: u64, what: &str) -> Result<&mut Seq> {
        self.seqs
            .get_mut(&seq_id)
            .ok_or_else(|| anyhow!("qwen4_exp runner: {what} on unknown seq_id {seq_id}"))
    }

    /// One decode step with the pick left to the caller, over the host logits.
    pub(crate) fn decode_step_pick(
        &mut self,
        seq_id: u64,
        last_token: u32,
        pick: &mut dyn FnMut(&mut [f32]) -> Result<u32>,
    ) -> Result<(u32, usize)> {
        let model = &self.model;
        let seq = self.seqs.get_mut(&seq_id).ok_or_else(|| {
            anyhow!("qwen4_exp runner: decode_step_pick on unknown seq_id {seq_id}")
        })?;
        let logits = model.forward(&[last_token], &mut seq.cache, true)?;
        let mut host = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(&logits)
            .context("qwen4_exp: logits to host")?;
        let next = pick(&mut host).context("qwen4_exp: sampler callback")?;
        seq.position += 1;
        Ok((next, seq.position))
    }

    pub(crate) fn decode_step_masked(
        &mut self,
        seq_id: u64,
        last_token: u32,
        mask: &mut dyn FnMut(&mut [f32]) -> Result<()>,
    ) -> Result<(u32, usize)> {
        self.decode_step_pick(seq_id, last_token, &mut |logits| {
            mask(logits)?;
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("qwen4_exp: empty logits"))
        })
    }
}

impl Qwen4ExpRunner {
    /// Write `seq_id`'s rollback point (the state after `prefix`, placed on a prefill grid of
    /// `chunk`) to the disk tier. Returns whether anything was written.
    pub(crate) fn persist_point(
        &mut self,
        seq_id: u64,
        prefix: &[u32],
        chunk: usize,
    ) -> Result<bool> {
        let Some(disk) = self.disk.as_ref() else {
            return Ok(false);
        };
        let key = crate::kv_disk::point_key(chunk, prefix);
        if disk.contains(&key) {
            return Ok(false);
        }
        let Some(seq) = self.seqs.get(&seq_id) else {
            return Ok(false);
        };
        let Some((point, position)) = seq.rollback.as_ref().filter(|(_, p)| *p == prefix.len())
        else {
            return Ok(false);
        };
        let _ = position;
        let t_read = std::time::Instant::now();
        let (layers, records) = point.to_records()?;
        let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;
        let manifest = KvManifest {
            model_fingerprint: self.fingerprint.clone(),
            created_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            position: prefix.len(),
            prefix_tokens: prefix.to_vec(),
            last_token: None,
            is_deep: true,
            layers,
        };
        let bytes: usize = records.iter().map(|r| r.data.len()).sum();
        let disk = self
            .disk
            .as_mut()
            .expect("disk tier present (checked above)");
        disk.put(&key, &manifest, &records)?;
        let superseded = self
            .seqs
            .get_mut(&seq_id)
            .and_then(|s| s.persisted_key.replace(key.clone()))
            .filter(|old| *old != key);
        if let Some(old) = superseded {
            disk.remove(&old)?;
        }
        eprintln!(
            "[qwen4_exp] kv-disk: point at {} = {:.0} MB (readback {read_ms:.0} ms)",
            prefix.len(),
            bytes as f64 / 1e6
        );
        Ok(true)
    }

    /// Start `seq_id` from the longest persisted state `prompt` continues (on this
    /// prompt's own prefill grid), with a rollback point there. Returns its length.
    pub(crate) fn restore_point(&mut self, seq_id: u64, prompt: &[u32]) -> Result<Option<usize>> {
        if self.disk.is_none() || self.seqs.contains_key(&seq_id) {
            return Ok(None);
        }
        let chunk = self.chunk.max(1);
        let last = prompt.len().saturating_sub(1) / chunk * chunk;
        for at in (chunk..=last).rev().step_by(chunk) {
            let key = crate::kv_disk::point_key(chunk, &prompt[..at]);
            let disk = self
                .disk
                .as_mut()
                .expect("disk tier present (checked above)");
            if !disk.contains(&key) {
                continue;
            }
            let Some((manifest, records)) = disk.get(&key)? else {
                continue;
            };
            if manifest.position != at || manifest.prefix_tokens != prompt[..at] {
                continue;
            }
            let cache =
                match self
                    .model
                    .cache_from_records(&manifest.layers, &records, &prompt[..at])
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "[qwen4_exp] kv-disk: state at {at} failed to load ({e:#}); skipping"
                        );
                        continue;
                    }
                };
            self.seqs.insert(
                seq_id,
                Seq {
                    rollback: Some((cache.clone(), at)),
                    cache,
                    position: at,
                    persisted_key: Some(key),
                },
            );
            return Ok(Some(at));
        }
        Ok(None)
    }
}

impl Runner for Qwen4ExpRunner {
    fn name(&self) -> &'static str {
        "qwen4_exp"
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        let mut cache = self.model.make_cache();
        let logits = self.forward_chunked(tokens, &mut cache)?;
        let next = Self::argmax(&logits)?;
        self.seqs.insert(
            seq_id,
            Seq {
                cache,
                position: tokens.len(),
                rollback: None,
                persisted_key: None,
            },
        );
        Ok((next, tokens.len()))
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        _position: usize,
    ) -> Result<(u32, usize)> {
        let model = &self.model;
        let seq = self
            .seqs
            .get_mut(&seq_id)
            .ok_or_else(|| anyhow!("qwen4_exp runner: decode_step on unknown seq_id {seq_id}"))?;
        let logits = model.forward(&[last_token], &mut seq.cache, true)?;
        let next = Self::argmax(&logits)?;
        seq.position += 1;
        Ok((next, seq.position))
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        self.seqs.remove(&seq_id);
        let _ = crate::metal_memory::clear_cache();
        Ok(())
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        // Feed a clone: a failed forward leaves the sequence exactly as it was.
        let mut cache = self.seq(seq_id, "extend")?.cache.clone();
        let logits = self.forward_chunked(tokens, &mut cache)?;
        let next = Self::argmax(&logits)?;
        let seq = self.seq(seq_id, "extend")?;
        seq.cache = cache;
        seq.position += tokens.len();
        Ok((next, seq.position))
    }

    fn forward_probe(&mut self, _seq_id: u64, _tokens: &[u32]) -> Result<ProbeRows> {
        bail!("qwen4_exp runner has no forward probe")
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        Ok(self.snapshot_state_deep(seq_id)?.0)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        let (cache, position) = self
            .snapshots
            .get(&snapshot_id)
            .cloned()
            .ok_or_else(|| anyhow!("qwen4_exp runner: unknown snapshot {snapshot_id}"))?;
        let seq = self.seq(seq_id, "restore_state")?;
        seq.cache = cache;
        seq.position = position;
        Ok(position)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        self.snapshots.remove(&snapshot_id);
        Ok(())
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        let seq = self.seq(seq_id, "snapshot_state_deep")?;
        let copy = (seq.cache.clone(), seq.position);
        let position = copy.1;
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.insert(id, copy);
        Ok((id, position))
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        let (cache, position) = self
            .snapshots
            .get(&snapshot_id)
            .cloned()
            .ok_or_else(|| anyhow!("qwen4_exp runner: unknown snapshot {snapshot_id}"))?;
        self.seqs.insert(
            dst_seq_id,
            Seq {
                cache,
                position,
                rollback: None,
                persisted_key: None,
            },
        );
        Ok(position)
    }

    fn prefill_grid(&self, _n: usize) -> Option<PrefillGrid> {
        Some(PrefillGrid { chunk: self.chunk })
    }

    fn mark_rollback(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.seq(seq_id, "mark_rollback")?;
        seq.rollback = Some((seq.cache.clone(), seq.position));
        Ok(())
    }

    fn rollback(&mut self, seq_id: u64) -> Result<usize> {
        let seq = self.seq(seq_id, "rollback")?;
        let (cache, position) = seq
            .rollback
            .clone()
            .ok_or_else(|| anyhow!("qwen4_exp runner: rollback without a mark on seq {seq_id}"))?;
        seq.cache = cache;
        seq.position = position;
        Ok(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_is_what_the_working_set_leaves_after_the_trunk_and_a_chunk() {
        // 36 GB Mac: 30.2 GB working set x 0.88. trunk + chunk reserve (3.6 + 2 + 1.1 GB).
        let (chunk, pool) = plan_memory(26_600_000_000, None);
        assert_eq!(chunk, 1024);
        assert!((19_500_000_000..20_500_000_000).contains(&pool), "{pool}");
        // 24 GB Mac: ~16 GB working set x 0.88.
        let (_, pool) = plan_memory(14_100_000_000, None);
        assert!(pool > 6_000_000_000, "{pool}");
        // Too little memory: the pool is zero and the slot floor takes over.
        assert_eq!(plan_memory(4_000_000_000, None).1, 0);
    }

    #[test]
    fn an_explicit_chunk_is_honoured_and_changes_the_pool() {
        let (chunk, smaller) = plan_memory(26_600_000_000, Some(4096));
        assert_eq!(chunk, 4096);
        assert!(smaller < plan_memory(26_600_000_000, None).1);
    }

    fn tiny_runner(chunk: usize) -> Qwen4ExpRunner {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny");
        Qwen4ExpRunner {
            model: Qwen4ExpModel::load(&dir, None).expect("tiny replica"),
            seqs: HashMap::new(),
            snapshots: HashMap::new(),
            next_snapshot: 0,
            chunk,
            disk: None,
            fingerprint: String::new(),
        }
    }

    fn decode_some(r: &mut Qwen4ExpRunner, seq: u64, mut last: u32, steps: usize) -> Vec<u32> {
        let mut out = vec![last];
        for _ in 0..steps {
            let (next, _) = r.decode_step(seq, last, 0).unwrap();
            out.push(next);
            last = next;
        }
        out
    }

    /// The whole-prompt marked feed is the cut-by-cut feed the session would otherwise do:
    /// same next token and position, and the same state at the rollback point and at the end.
    #[test]
    fn a_marked_feed_in_one_call_equals_feeding_cut_by_cut() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny/reference/outputs.safetensors");
        let want = Array::load_safetensors(dir).expect("reference outputs");
        let ids: Vec<u32> = want["ids"]
            .as_slice::<i32>()
            .iter()
            .map(|&t| t as u32)
            .collect();
        assert!(ids.len() >= 48);
        let ids = &ids[..48];

        // Cold: chunks of 16, rollback point at 32.
        let (mut one, mut cuts) = (tiny_runner(16), tiny_runner(16));
        let a = one
            .feed_with_mark(1, true, ids, 32)
            .unwrap()
            .expect("supported");
        cuts.prefill(1, &ids[..16]).unwrap();
        cuts.extend(1, &ids[16..32]).unwrap();
        cuts.mark_rollback(1).unwrap();
        let b = cuts.extend(1, &ids[32..]).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            decode_some(&mut one, 1, a.0, 5),
            decode_some(&mut cuts, 1, b.0, 5)
        );
        assert_eq!(one.rollback(1).unwrap(), 32);
        assert_eq!(cuts.rollback(1).unwrap(), 32);
        let first = ids[32];
        assert_eq!(
            decode_some(&mut one, 1, first, 5),
            decode_some(&mut cuts, 1, first, 5)
        );

        // Continuing a sequence that stands at 20 (off the grid): the pieces end on the grid.
        let (mut one, mut cuts) = (tiny_runner(16), tiny_runner(16));
        for r in [&mut one, &mut cuts] {
            r.prefill(2, &ids[..20]).unwrap();
        }
        let a = one
            .feed_with_mark(2, false, &ids[20..], 12)
            .unwrap()
            .expect("supported");
        cuts.extend(2, &ids[20..32]).unwrap();
        cuts.mark_rollback(2).unwrap();
        let b = cuts.extend(2, &ids[32..]).unwrap();
        assert_eq!(a, b);
        assert_eq!(one.rollback(2).unwrap(), 32);
        assert_eq!(cuts.rollback(2).unwrap(), 32);
        assert_eq!(
            decode_some(&mut one, 2, ids[32], 5),
            decode_some(&mut cuts, 2, ids[32], 5)
        );

        // A mark that is not on the grid is declined, not guessed at.
        let mut r = tiny_runner(16);
        assert!(r.feed_with_mark(3, true, ids, 30).unwrap().is_none());
    }

    #[test]
    fn chunks_end_on_the_absolute_grid() {
        assert_eq!(chunk_cuts(0, 40, 16), vec![16, 32, 40]);
        assert_eq!(chunk_cuts(20, 28, 16), vec![12, 28]);
        assert_eq!(chunk_cuts(16, 16, 16), vec![16]);
        assert_eq!(chunk_cuts(0, 5, 16), vec![5]);
    }
}
