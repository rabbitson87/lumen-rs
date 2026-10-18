//! The n-gram embedding table: ~32 GB of 4-bit rows, 16 of which are read per
//! token. Rows come from shards that are either held as MLX arrays (small
//! checkpoints and tests) or read from the safetensors files on demand, one
//! ~100-byte row at a time, with a cache of recently used rows in front.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::{Array, Dtype};

use crate::native_quant::{MODE_AFFINE, dequantize_with_mode};
use crate::offload::{IoMode, IoPool, ReadJob, read_header};

/// One shard of the table as stored: packed 4-bit rows plus per-group scales
/// and biases.
pub struct ShardArrays {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

/// Rows of the sharded table, dequantized to `[n_rows, dim]` on request.
pub enum NgramTable {
    /// Shards held as MLX arrays.
    Resident(ResidentTable),
    /// Rows read from the checkpoint's files on demand.
    File(FileTable),
}

impl NgramTable {
    /// Dequantized rows for `row_ids`, in the same order: `[row_ids.len(), dim]`.
    pub fn gather(&self, row_ids: &[u64]) -> Result<Array> {
        if row_ids.is_empty() {
            bail!("n-gram gather of 0 rows");
        }
        match self {
            Self::Resident(t) => t.gather(row_ids),
            Self::File(t) => t.gather(row_ids),
        }
    }
}

// ---- resident ------------------------------------------------------------------------------

pub struct ResidentTable {
    shards: Vec<ShardArrays>,
    rows_per_shard: u64,
    group_size: i32,
    bits: i32,
}

impl ResidentTable {
    pub fn new(
        shards: Vec<ShardArrays>,
        rows_per_shard: u64,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        if shards.is_empty() {
            bail!("n-gram table has no shards");
        }
        if rows_per_shard == 0 {
            bail!("n-gram table shard has 0 rows");
        }
        Ok(Self {
            shards,
            rows_per_shard,
            group_size,
            bits,
        })
    }

    fn gather(&self, row_ids: &[u64]) -> Result<Array> {
        // Group the request by shard so each shard is indexed once, then put
        // the rows back in request order.
        let mut by_shard: Vec<(usize, Vec<i32>, Vec<usize>)> = Vec::new();
        let mut slot_of_shard = HashMap::new();
        for (pos, &id) in row_ids.iter().enumerate() {
            let shard = (id / self.rows_per_shard) as usize;
            let row = (id % self.rows_per_shard) as i32;
            if shard >= self.shards.len() {
                bail!(
                    "n-gram row {id} is in shard {shard}; the table has {}",
                    self.shards.len()
                );
            }
            let slot = *slot_of_shard.entry(shard).or_insert_with(|| {
                by_shard.push((shard, Vec::new(), Vec::new()));
                by_shard.len() - 1
            });
            by_shard[slot].1.push(row);
            by_shard[slot].2.push(pos);
        }

        let mut parts: Vec<Array> = Vec::with_capacity(by_shard.len());
        let mut order: Vec<usize> = Vec::with_capacity(row_ids.len());
        for (shard, rows, positions) in &by_shard {
            let idx = Array::from_slice(rows, &[rows.len() as i32]);
            let s = &self.shards[*shard];
            let take = |a: &Array, what: &str| -> Result<Array> {
                take_axis(a, &idx, 0)
                    .with_context(|| format!("n-gram shard {shard}: take {what} rows"))
            };
            let deq = dequantize_with_mode(
                &take(&s.weight, "weight")?,
                &take(&s.scales, "scales")?,
                Some(&take(&s.biases, "biases")?),
                self.group_size,
                self.bits,
                MODE_AFFINE,
            )
            .with_context(|| format!("n-gram shard {shard}: dequantize"))?;
            parts.push(deq);
            order.extend_from_slice(positions);
        }
        if parts.len() == 1 && order.iter().enumerate().all(|(i, &p)| i == p) {
            return Ok(parts.remove(0));
        }
        let refs: Vec<&Array> = parts.iter().collect();
        let joined = mlx_rs::ops::concatenate_axis(&refs, 0).context("n-gram: join shards")?;
        // joined[j] holds request position order[j]; invert to request order.
        let mut inverse = vec![0i32; order.len()];
        for (j, &pos) in order.iter().enumerate() {
            inverse[pos] = j as i32;
        }
        let inv = Array::from_slice(&inverse, &[inverse.len() as i32]);
        take_axis(&joined, &inv, 0).map_err(|e| anyhow!("n-gram: restore request order: {e}"))
    }
}

// ---- file-backed ---------------------------------------------------------------------------

/// Where one shard's three tensors start: (index into `FileTable`'s file list,
/// absolute offset of row 0). A build may put a shard's tensors in different files.
struct ShardAt {
    weight: (usize, u64),
    scales: (usize, u64),
    biases: (usize, u64),
}

/// Recently read rows, `[weight | scales | biases]` bytes each. The least
/// recently used quarter goes when the cache is full.
struct RowCache {
    rows: HashMap<u64, (Box<[u8]>, u64)>,
    capacity: usize,
    clock: u64,
}

impl RowCache {
    fn get(&mut self, id: u64) -> Option<&[u8]> {
        self.clock += 1;
        let now = self.clock;
        self.rows.get_mut(&id).map(|(bytes, used)| {
            *used = now;
            &bytes[..]
        })
    }

    fn insert(&mut self, id: u64, bytes: Box<[u8]>) {
        if self.rows.len() >= self.capacity {
            let mut ages: Vec<u64> = self.rows.values().map(|(_, used)| *used).collect();
            ages.sort_unstable();
            let cutoff = ages[ages.len() / 4];
            self.rows.retain(|_, (_, used)| *used > cutoff);
        }
        self.clock += 1;
        self.rows.insert(id, (bytes, self.clock));
    }
}

/// Cumulative counts, for logging and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct RowStats {
    pub requested: u64,
    pub read_from_disk: u64,
}

pub struct FileTable {
    pool: IoPool,
    shards: Vec<ShardAt>,
    rows_per_shard: u64,
    /// Bytes of one row's weight, scales and biases.
    w_bytes: usize,
    s_bytes: usize,
    packed_cols: i32,
    scale_cols: i32,
    scales_dtype: Dtype,
    group_size: i32,
    bits: i32,
    cache: Mutex<RowCache>,
    stats: Mutex<RowStats>,
}

/// Rows kept after use: 400,000 rows is ~40 MB of the shipped table.
pub const DEFAULT_ROW_CACHE: usize = 400_000;
const LANES: usize = 16;

impl FileTable {
    /// Locate `shard_name(i)` (`<prefix>.weight` / `.scales` / `.biases`) for
    /// `i` in `0..n_shards` in the safetensors files of `dir`.
    pub(crate) fn open(
        dir: &Path,
        n_shards: usize,
        shard_name: &dyn Fn(usize) -> String,
        group_size: i32,
        bits: i32,
        row_cache: usize,
        mode: IoMode,
    ) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("n-gram table: read_dir({})", dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();

        let wanted: HashMap<String, (usize, &'static str)> = (0..n_shards)
            .flat_map(|i| {
                let base = shard_name(i);
                ["weight", "scales", "biases"]
                    .into_iter()
                    .map(move |part| (format!("{base}.{part}"), (i, part)))
            })
            .collect();
        let mut found: Vec<[Option<(usize, u64, Vec<i64>, String)>; 3]> =
            (0..n_shards).map(|_| [None, None, None]).collect();
        for (file_idx, path) in files.iter().enumerate() {
            for t in read_header(path)? {
                if let Some(&(shard, part)) = wanted.get(&t.name) {
                    let slot = match part {
                        "weight" => 0,
                        "scales" => 1,
                        _ => 2,
                    };
                    found[shard][slot] = Some((file_idx, t.offset(), t.shape.clone(), t.dtype));
                }
            }
        }

        let mut shards = Vec::with_capacity(n_shards);
        let mut layout: Option<(u64, i32, i32, String)> = None;
        for (i, parts) in found.into_iter().enumerate() {
            let [Some(w), Some(s), Some(b)] = parts else {
                bail!(
                    "n-gram table: shard {i} is missing a tensor in {}",
                    dir.display()
                );
            };
            let (rows, wc) = (w.2[0] as u64, w.2[1] as i32);
            let sc = s.2[1] as i32;
            if s.2[0] as u64 != rows || b.2[0] as u64 != rows || b.2[1] as i32 != sc {
                bail!("n-gram table: shard {i} has inconsistent row counts");
            }
            match &layout {
                None => layout = Some((rows, wc, sc, s.3.clone())),
                Some((r, c, g, d)) if *r == rows && *c == wc && *g == sc && *d == s.3 => {}
                Some(_) => bail!("n-gram table: shard {i} has a different layout from shard 0"),
            }
            shards.push(ShardAt {
                weight: (w.0, w.1),
                scales: (s.0, s.1),
                biases: (b.0, b.1),
            });
        }
        let (rows_per_shard, packed_cols, scale_cols, sdtype) =
            layout.ok_or_else(|| anyhow!("n-gram table has no shards"))?;
        let (scales_dtype, elem) = match sdtype.as_str() {
            "BF16" => (Dtype::Bfloat16, 2usize),
            "F16" => (Dtype::Float16, 2),
            "F32" => (Dtype::Float32, 4),
            other => bail!("n-gram table: scales have dtype {other}"),
        };
        Ok(Self {
            pool: IoPool::new(&files, LANES, mode)?,
            shards,
            rows_per_shard,
            w_bytes: packed_cols as usize * 4,
            s_bytes: scale_cols as usize * elem,
            packed_cols,
            scale_cols,
            scales_dtype,
            group_size,
            bits,
            cache: Mutex::new(RowCache {
                rows: HashMap::new(),
                capacity: row_cache.max(1),
                clock: 0,
            }),
            stats: Mutex::new(RowStats::default()),
        })
    }

    pub fn stats(&self) -> RowStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn row_bytes(&self) -> usize {
        self.w_bytes + 2 * self.s_bytes
    }

    fn gather(&self, row_ids: &[u64]) -> Result<Array> {
        let rb = self.row_bytes();
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());

        // Distinct rows this call has to fetch.
        let mut missing: Vec<u64> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &id in row_ids {
            if (id / self.rows_per_shard) as usize >= self.shards.len() {
                bail!(
                    "n-gram row {id} is in shard {}; the table has {}",
                    id / self.rows_per_shard,
                    self.shards.len()
                );
            }
            if cache.get(id).is_none() && seen.insert(id) {
                missing.push(id);
            }
        }

        if !missing.is_empty() {
            let mut buf = vec![0u8; missing.len() * rb];
            let base = buf.as_mut_ptr();
            let mut jobs = Vec::with_capacity(missing.len() * 3);
            for (n, &id) in missing.iter().enumerate() {
                let shard = &self.shards[(id / self.rows_per_shard) as usize];
                let row = id % self.rows_per_shard;
                // SAFETY: `n * rb + rb <= buf.len()`; the three slices of a row do not overlap.
                let dst = unsafe { base.add(n * rb) };
                jobs.push(ReadJob {
                    file: shard.weight.0,
                    offset: shard.weight.1 + row * self.w_bytes as u64,
                    len: self.w_bytes,
                    dst,
                });
                jobs.push(ReadJob {
                    file: shard.scales.0,
                    offset: shard.scales.1 + row * self.s_bytes as u64,
                    len: self.s_bytes,
                    dst: unsafe { dst.add(self.w_bytes) },
                });
                jobs.push(ReadJob {
                    file: shard.biases.0,
                    offset: shard.biases.1 + row * self.s_bytes as u64,
                    len: self.s_bytes,
                    dst: unsafe { dst.add(self.w_bytes + self.s_bytes) },
                });
            }
            // SAFETY: every destination is a distinct slice of `buf`, which outlives the
            // call and is not touched until it returns.
            unsafe { self.pool.run(jobs) }.context("n-gram table: read rows")?;
            for (n, &id) in missing.iter().enumerate() {
                cache.insert(id, buf[n * rb..(n + 1) * rb].to_vec().into_boxed_slice());
            }
        }
        {
            let mut st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
            st.requested += row_ids.len() as u64;
            st.read_from_disk += missing.len() as u64;
        }

        // Rows in request order, one buffer per tensor.
        let n = row_ids.len();
        let mut w = Vec::with_capacity(n * self.w_bytes);
        let mut s = Vec::with_capacity(n * self.s_bytes);
        let mut b = Vec::with_capacity(n * self.s_bytes);
        for &id in row_ids {
            let row = cache
                .get(id)
                .ok_or_else(|| anyhow!("n-gram row {id} vanished from the cache"))?;
            w.extend_from_slice(&row[..self.w_bytes]);
            s.extend_from_slice(&row[self.w_bytes..self.w_bytes + self.s_bytes]);
            b.extend_from_slice(&row[self.w_bytes + self.s_bytes..]);
        }
        drop(cache);
        let n = n as i32;
        // SAFETY: each buffer holds exactly `n * cols * element size` bytes of the dtype named.
        let (weight, scales, biases) = unsafe {
            (
                Array::from_raw_data(w.as_ptr().cast(), &[n, self.packed_cols], Dtype::Uint32),
                Array::from_raw_data(s.as_ptr().cast(), &[n, self.scale_cols], self.scales_dtype),
                Array::from_raw_data(b.as_ptr().cast(), &[n, self.scale_cols], self.scales_dtype),
            )
        };
        dequantize_with_mode(
            &weight,
            &scales,
            Some(&biases),
            self.group_size,
            self.bits,
            MODE_AFFINE,
        )
        .context("n-gram table: dequantize")
    }
}
