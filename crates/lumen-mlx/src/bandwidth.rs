//! How fast this Mac's GPU can stream memory.
//!
//! Decode of a model that does not fit in RAM costs a fixed part per token (the GPU reading the
//! trunk and the experts it already holds) plus the SSD reads for the experts it does not. The
//! fixed part scales with memory bandwidth, so the app measures it here, once, and uses it with
//! the SSD speed to estimate decode for this Mac (`lumen-server --measure-memory-bandwidth`).
//!
//! The probe sums a 1 GiB array of random floats, which reads every byte exactly once and
//! writes one scalar. The array is filled with random numbers because a constant or a range
//! would be a lazy broadcast that reads nothing. The bytes are the same ones a matmul over a
//! weight matrix streams, so the number is what the GPU gets on that access pattern, not the
//! chip's data-sheet peak.

use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::ops::sum_axis;
use mlx_rs::random::uniform;
use mlx_rs::transforms::eval;

/// Bytes the probe reads per pass.
const PROBE_BYTES: usize = 1 << 30;
/// Passes thrown away so the first kernel launches and page-ins do not count.
const WARMUP_PASSES: usize = 3;
/// Passes timed.
const TIMED_PASSES: usize = 15;

/// What one probe run saw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryBandwidth {
    /// Median over the timed passes, GB/s (1 GB = 10^9 bytes).
    pub median_gbps: f64,
    /// Fastest pass, GB/s.
    pub best_gbps: f64,
    /// Bytes read per pass.
    pub bytes: u64,
    pub passes: usize,
}

impl MemoryBandwidth {
    /// One line of JSON, what `lumen-server --measure-memory-bandwidth` prints.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"gpu_read_gbps\":{:.1},\"best_gbps\":{:.1},\"bytes\":{},\"passes\":{}}}",
            self.median_gbps, self.best_gbps, self.bytes, self.passes
        )
    }
}

/// GB/s for `bytes` read in `seconds`; 0 for a non-positive time.
fn gbps(bytes: u64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        bytes as f64 / seconds / 1e9
    } else {
        0.0
    }
}

/// Median of `samples` (sorted in place); 0 when empty.
fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2.0
    }
}

/// Measure the GPU's read bandwidth. Takes about a second and 1 GiB of memory.
pub fn measure_memory_bandwidth() -> Result<MemoryBandwidth> {
    let elements = (PROBE_BYTES / std::mem::size_of::<f32>()) as i32;
    let data = uniform::<_, f32>(0.0f32, 1.0f32, &[elements], None).context("fill the probe")?;
    eval([&data]).context("fill the probe")?;

    let mut seconds = Vec::with_capacity(TIMED_PASSES);
    for pass in 0..WARMUP_PASSES + TIMED_PASSES {
        let started = Instant::now();
        let total = sum_axis(&data, 0, false).context("sum the probe")?;
        eval([&total]).context("sum the probe")?;
        if pass >= WARMUP_PASSES {
            seconds.push(started.elapsed().as_secs_f64());
        }
    }
    let bytes = PROBE_BYTES as u64;
    let mut rates: Vec<f64> = seconds.iter().map(|&s| gbps(bytes, s)).collect();
    let best_gbps = rates.iter().copied().fold(0.0, f64::max);
    Ok(MemoryBandwidth {
        median_gbps: median(&mut rates),
        best_gbps,
        bytes,
        passes: seconds.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_of_an_odd_and_an_even_count() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
    }

    #[test]
    fn a_rate_is_bytes_over_seconds_in_decimal_gigabytes() {
        assert_eq!(gbps(2_000_000_000, 0.5), 4.0);
        assert_eq!(gbps(1, 0.0), 0.0);
    }

    #[test]
    fn the_json_line_has_the_keys_the_app_reads() {
        let line = MemoryBandwidth {
            median_gbps: 231.46,
            best_gbps: 240.04,
            bytes: 1 << 30,
            passes: 15,
        }
        .to_json();
        assert_eq!(
            line,
            "{\"gpu_read_gbps\":231.5,\"best_gbps\":240.0,\"bytes\":1073741824,\"passes\":15}"
        );
    }

    /// Needs a Metal device: the number is whatever this GPU does, so only its sanity is checked
    /// (a 2020+ Mac streams more than 20 GB/s and nothing streams more than a TB/s).
    #[test]
    #[ignore = "tier 1: needs a Metal device"]
    fn the_probe_reports_a_plausible_rate() {
        let m = measure_memory_bandwidth().unwrap();
        assert!((20.0..1000.0).contains(&m.median_gbps), "{m:?}");
        assert!(m.best_gbps >= m.median_gbps);
        assert_eq!(m.passes, TIMED_PASSES);
    }
}
