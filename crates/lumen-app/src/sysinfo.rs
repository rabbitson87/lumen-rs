//! System probe — installed RAM, arch, etc. Used to seed sensible
//! defaults for the SERVER card's memory caps and to flag recommended
//! models that won't fit.
//!
//! We don't depend on a heavy crate (`sysinfo`, `mach2`) for the RAM
//! probe — `sysctl hw.memsize` is one syscall via libc and matches what
//! Activity Monitor shows.

use std::process::Command;

use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SystemInfo {
    /// Total installed RAM in GB (rounded).
    pub ram_gb: usize,
    /// CPU architecture as reported by `std::env::consts::ARCH` (`aarch64` /
    /// `x86_64`).
    pub arch: &'static str,
    /// Memory caps the UI / config will use as defaults.
    pub recommended: MemoryDefaults,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MemoryDefaults {
    pub wired_limit_gb: usize,
    pub cache_limit_gb: usize,
    pub memory_limit_gb: usize,
}

/// Total installed RAM in bytes from `sysctl hw.memsize`. Returns `None`
/// on non-Darwin systems or if the syscall fails.
pub fn total_ram_bytes() -> Option<u64> {
    let out = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    s.trim().parse().ok()
}

pub fn total_ram_gb() -> Option<usize> {
    total_ram_bytes().map(|b| (b / (1024 * 1024 * 1024)) as usize)
}

/// RAM-aware Metal memory caps. Apple Silicon uses unified memory, so the
/// numbers below are fractions of total system RAM:
///
/// - `wired_limit`  — 70%: page-locked weight residency ceiling. Too high
///   risks the OS killing other apps; too low triggers GPU evictions during
///   decode (2-5× regression).
/// - `cache_limit`  — flat 2 GB: MLX's transient buffer reuse pool.
///   Doesn't scale with system RAM — the pool only needs to hold a handful
///   of activation/scratch buffers between requests.
/// - `memory_limit` — 85%: soft total cap. MLX evicts cache more aggressively
///   above this before hitting the hard wired ceiling.
pub fn ram_defaults(total_gb: usize) -> MemoryDefaults {
    let wired = (total_gb * 70 / 100).max(2);
    let cache = 2;
    let memory = (total_gb * 85 / 100).max(4);
    MemoryDefaults {
        wired_limit_gb: wired,
        cache_limit_gb: cache,
        memory_limit_gb: memory,
    }
}

/// Snapshot of current system memory pressure, suitable for a live UI
/// monitor. `used_bytes` mirrors Activity Monitor's "Memory Used" reading
/// (wired + active + compressor-occupied pages — i.e. what's actually
/// resident vs. freeable cache). Returns `None` if `vm_stat` can't be
/// parsed (non-Darwin / format drift across macOS versions).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MemoryUsage {
    pub used_bytes: u64,
    pub total_bytes: u64,
    /// Wired (kernel-locked, non-pageable) bytes. Everything else in
    /// `used_bytes` — active working sets and the compressor's pages — is
    /// reclaimable under pressure (paged out / swap-compressed) when a large
    /// MLX allocation arrives, so `total - wired` is the realistic ceiling a
    /// model can claim, whereas `total - used` is the no-swap comfortable free.
    pub wired_bytes: u64,
}

pub fn current_memory_usage() -> Option<MemoryUsage> {
    let total = total_ram_bytes()?;
    let out = Command::new("vm_stat").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;

    // First line: "Mach Virtual Memory Statistics: (page size of N bytes)"
    let mut page_size: u64 = 16384;
    let mut wired: u64 = 0;
    let mut active: u64 = 0;
    let mut compressed: u64 = 0;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            if let Some(num) = rest.split_whitespace().next() {
                if let Ok(n) = num.parse() {
                    page_size = n;
                }
            }
            continue;
        }
        let (label, val) = match line.split_once(':') {
            Some(pair) => pair,
            None => continue,
        };
        let val_pages: u64 = match val.trim().trim_end_matches('.').parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        match label.trim() {
            "Pages wired down" => wired = val_pages,
            "Pages active" => active = val_pages,
            "Pages occupied by compressor" => compressed = val_pages,
            _ => {}
        }
    }
    let used_bytes = (wired + active + compressed) * page_size;
    Some(MemoryUsage {
        used_bytes,
        total_bytes: total,
        wired_bytes: wired * page_size,
    })
}

pub fn probe() -> SystemInfo {
    // Fall back to a conservative 16 GB assumption if the syscall fails. We'd
    // rather over-evict than starve the OS — 16 GB is the floor for any
    // shippable model.
    let ram_gb = total_ram_gb().unwrap_or(16);
    SystemInfo {
        ram_gb,
        arch: std::env::consts::ARCH,
        recommended: ram_defaults(ram_gb),
    }
}
