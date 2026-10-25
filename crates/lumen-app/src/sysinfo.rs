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
    /// What the rest of the machine gives up without anyone being compressed or swapped: free,
    /// file-backed and purgeable pages. The number Flash-Next's server plans its expert pool
    /// within at load (`host_memory` in lumen-mlx reads the same three counts).
    pub reclaimable_bytes: u64,
    /// The running lumen-server's footprint (0 when none runs): memory a server started in its
    /// place would have too.
    pub server_footprint_bytes: u64,
}

/// The page counts of `vm_stat`'s output this monitor reads, in pages.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct VmStat {
    page_size: u64,
    free: u64,
    active: u64,
    wired: u64,
    purgeable: u64,
    file_backed: u64,
    compressor: u64,
}

fn parse_vm_stat(s: &str) -> VmStat {
    // First line: "Mach Virtual Memory Statistics: (page size of N bytes)"
    let mut vm = VmStat {
        page_size: 16384,
        ..VmStat::default()
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            if let Some(num) = rest.split_whitespace().next() {
                if let Ok(n) = num.parse() {
                    vm.page_size = n;
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
            "Pages free" => vm.free = val_pages,
            "Pages active" => vm.active = val_pages,
            "Pages wired down" => vm.wired = val_pages,
            "Pages purgeable" => vm.purgeable = val_pages,
            "File-backed pages" => vm.file_backed = val_pages,
            "Pages occupied by compressor" => vm.compressor = val_pages,
            _ => {}
        }
    }
    vm
}

/// The physical footprint of process `pid` (what Activity Monitor calls its memory), for a
/// process this user owns.
pub fn process_footprint(pid: u32) -> Option<u64> {
    // SAFETY: an all-zero `rusage_info_v2` is a valid value of a plain C struct.
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is the struct the V2 flavor fills; the kernel writes at most its size.
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            (&raw mut info).cast::<libc::rusage_info_t>(),
        )
    };
    (rc == 0).then_some(info.ri_phys_footprint)
}

/// The memory monitor's snapshot; `server_pid` is the running lumen-server's, if one runs.
pub fn current_memory_usage(server_pid: Option<u32>) -> Option<MemoryUsage> {
    let total = total_ram_bytes()?;
    let out = Command::new("vm_stat").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let vm = parse_vm_stat(&String::from_utf8(out.stdout).ok()?);
    let page = vm.page_size;
    Some(MemoryUsage {
        used_bytes: (vm.wired + vm.active + vm.compressor) * page,
        total_bytes: total,
        wired_bytes: vm.wired * page,
        reclaimable_bytes: (vm.free + vm.file_backed + vm.purgeable) * page,
        server_footprint_bytes: server_pid.and_then(process_footprint).unwrap_or(0),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `vm_stat` on a 36 GB Mac with an editor, a browser and chat apps open.
    const SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   640613.
Pages active:                                 645980.
Pages inactive:                               649963.
Pages speculative:                              3561.
Pages throttled:                                   0.
Pages wired down:                             214120.
Pages purgeable:                               19196.
\"Translation faults\":                     9071745168.
File-backed pages:                            421392.
Anonymous pages:                              878112.
Pages stored in compressor:                   305342.
Pages occupied by compressor:                 147990.
";

    #[test]
    fn free_file_backed_and_purgeable_pages_are_what_a_newcomer_gets_without_swapping() {
        let vm = parse_vm_stat(SAMPLE);
        assert_eq!(
            vm,
            VmStat {
                page_size: 16384,
                free: 640_613,
                active: 645_980,
                wired: 214_120,
                purgeable: 19_196,
                file_backed: 421_392,
                compressor: 147_990,
            }
        );
        // Speculative pages are file pages read ahead, already among the file-backed ones;
        // inactive and anonymous pages belong to other apps.
        assert_eq!(
            (vm.free + vm.file_backed + vm.purgeable) * vm.page_size,
            17_714_397_184
        );
    }

    #[test]
    fn this_process_has_a_footprint() {
        assert!(process_footprint(std::process::id()).is_some_and(|b| b > 0));
    }
}
