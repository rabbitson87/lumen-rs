//! How much memory the rest of the machine would give this process right now, read from the
//! kernel's page counts (`host_statistics64(HOST_VM_INFO64)`, the numbers `vm_stat` prints).
//!
//! "Reclaimable" here is what the kernel can hand this process without compressing or swapping
//! anyone's memory:
//!
//! - free pages (`vm_stat`'s "Pages free");
//! - file-backed pages ("File-backed pages"): copies of files, which the kernel drops (writing a
//!   dirty one back to its file, not to swap) and reads again when they are needed. Speculative
//!   pages (read-ahead) are among them: anonymous plus file-backed pages are exactly the active,
//!   inactive and speculative ones;
//! - volatile purgeable pages ("Pages purgeable"): caches their owners marked discardable.
//!
//! Anonymous pages, inactive ones included, are not: they are another app's heap or GPU buffers,
//! and taking them means compressing them or writing them to swap, which is what reading this
//! is for avoiding. Neither are wired pages or the pages the compressor occupies.
//!
//! `memory_pressure`'s "System-wide memory free percentage" (`kern.memorystatus_level`) is a
//! different number: active + inactive + free + speculative pages over all of RAM, so every page
//! that is not wired or compressed, other apps' anonymous memory included. It says how far the
//! system is from compressing harder, not how much a newcomer can take without making it do so:
//! on a 36 GB Mac with an editor, a browser and chat apps open it read 51% (19.7 GB) while the
//! reclaimable pages were 8.3 GB.

/// The page counts a memory plan reads, in bytes, and this process's own footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostMemory {
    /// Installed RAM (`hw.memsize`).
    pub ram: u64,
    /// Free pages, speculative ones not included (`vm_stat`'s "Pages free").
    pub free: u64,
    /// Pages holding file contents, speculative read-ahead included ("File-backed pages").
    pub file_backed: u64,
    /// Volatile purgeable pages ("Pages purgeable").
    pub purgeable: u64,
    /// Anonymous pages of every process, this one's included ("Anonymous pages").
    pub anonymous: u64,
    /// Wired pages ("Pages wired down").
    pub wired: u64,
    /// Pages the compressor occupies ("Pages occupied by compressor").
    pub compressor: u64,
    /// This process's physical footprint (what Activity Monitor calls its memory): its anonymous
    /// and compressed pages and its GPU buffers, none of which is in [`Self::reclaimable`].
    pub own: u64,
}

impl HostMemory {
    /// The page counts now, or `None` where they cannot be read (any OS but macOS).
    pub fn read() -> Option<Self> {
        imp::read()
    }

    /// Bytes the rest of the machine gives up without anyone's memory being compressed or
    /// swapped: free, file-backed and purgeable pages.
    pub fn reclaimable(&self) -> u64 {
        self.free + self.file_backed + self.purgeable
    }

    /// Bytes this process could hold without that: what it holds, plus [`Self::reclaimable`].
    pub fn available_to_self(&self) -> u64 {
        self.own + self.reclaimable()
    }

    /// Bytes the rest of the machine holds that cannot be taken without compressing or swapping
    /// it: everything that is neither reclaimable nor this process's.
    pub fn held_elsewhere(&self) -> u64 {
        self.ram.saturating_sub(self.available_to_self())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::HostMemory;
    use std::sync::OnceLock;

    unsafe extern "C" {
        // `libc` declares it deprecated in favour of the `mach2` crate; the call is the same.
        fn mach_host_self() -> libc::mach_port_t;
    }

    /// The host port: one send right, taken once (each `mach_host_self` call adds a reference).
    fn host() -> libc::mach_port_t {
        static HOST: OnceLock<libc::mach_port_t> = OnceLock::new();
        // SAFETY: no arguments; returns this task's send right to the host port.
        *HOST.get_or_init(|| unsafe { mach_host_self() })
    }

    fn ram() -> Option<u64> {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: the name is NUL-terminated, `bytes` is a u64 and `len` says so.
        let rc = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && bytes > 0).then_some(bytes)
    }

    fn page_size() -> u64 {
        // SAFETY: no arguments.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 { p as u64 } else { 16_384 }
    }

    /// The physical footprint of process `pid` (`proc_pid_rusage`), for a process this user owns.
    fn footprint(pid: i32) -> Option<u64> {
        // SAFETY: an all-zero `rusage_info_v2` is a valid value of a plain C struct.
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is the struct the V2 flavor fills; the kernel writes at most its size.
        let rc = unsafe {
            libc::proc_pid_rusage(
                pid,
                libc::RUSAGE_INFO_V2,
                (&raw mut info).cast::<libc::rusage_info_t>(),
            )
        };
        (rc == 0).then_some(info.ri_phys_footprint)
    }

    pub(super) fn read() -> Option<HostMemory> {
        // SAFETY: an all-zero `vm_statistics64` is a valid value of a plain C struct.
        let mut vm: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `vm` is the struct HOST_VM_INFO64 fills and `count` is its size in integers.
        let kr = unsafe {
            libc::host_statistics64(
                host(),
                libc::HOST_VM_INFO64,
                (&raw mut vm).cast::<libc::integer_t>(),
                &mut count,
            )
        };
        if kr != libc::KERN_SUCCESS {
            return None;
        }
        let page = page_size();
        let bytes = |pages: u32| pages as u64 * page;
        // The struct is packed: copy each field out before using it.
        let (free_count, speculative) = (vm.free_count, vm.speculative_count);
        Some(HostMemory {
            ram: ram()?,
            // `free_count` counts the speculative pages too; `vm_stat` subtracts them, and they
            // are among the file-backed ones.
            free: bytes(free_count.saturating_sub(speculative)),
            file_backed: bytes(vm.external_page_count),
            purgeable: bytes(vm.purgeable_count),
            anonymous: bytes(vm.internal_page_count),
            wired: bytes(vm.wire_count),
            compressor: bytes(vm.compressor_page_count),
            // SAFETY: no arguments.
            own: footprint(unsafe { libc::getpid() }).unwrap_or(0),
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn read() -> Option<super::HostMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    /// `vm_stat` on the 36 GB Mac this was written on (16 KB pages), with an editor, a browser,
    /// chat apps and an idle model server open; the server's 10.7 GB sat in the compressor.
    fn busy_36gb() -> HostMemory {
        HostMemory {
            ram: 38_654_705_664,
            free: 92_438_528,
            file_backed: 8_125_726_720,
            purgeable: 42_876_928,
            anonymous: 11_263_918_080,
            wired: 4_838_866_944,
            compressor: 13_398_736_896,
            own: 3_000_000,
        }
    }

    #[test]
    fn free_file_backed_and_purgeable_pages_are_reclaimable_and_nothing_else() {
        let h = busy_36gb();
        assert_eq!(h.reclaimable(), 92_438_528 + 8_125_726_720 + 42_876_928);
        assert_eq!(h.available_to_self(), h.reclaimable() + 3_000_000);
        // The rest (anonymous, wired and compressed memory, and what the kernel does not count
        // in any of those) stays with whoever holds it.
        assert_eq!(
            h.held_elsewhere(),
            38_654_705_664 - 8_261_042_176 - 3_000_000
        );
        assert!(h.held_elsewhere() > h.anonymous + h.wired + h.compressor - GB);
    }

    #[test]
    fn a_footprint_larger_than_the_machine_holds_nothing_elsewhere() {
        let h = HostMemory {
            ram: 16 * GB,
            free: GB,
            own: 20 * GB,
            ..HostMemory::default()
        };
        assert_eq!(h.held_elsewhere(), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn this_mac_reports_counts_that_add_up() {
        let h = HostMemory::read().expect("host_statistics64 answers on macOS");
        assert!(h.ram > GB, "{h:?}");
        assert!(h.reclaimable() <= h.ram, "{h:?}");
        assert!(h.own > 0, "a running test process has a footprint: {h:?}");
        assert!(h.anonymous + h.file_backed + h.wired <= h.ram, "{h:?}");
    }

    /// `cargo test -p lumen-mlx --lib host_memory -- --ignored --nocapture`: this Mac's reading.
    #[test]
    #[ignore = "prints this machine's numbers"]
    fn print_this_mac() {
        let h = HostMemory::read().expect("macOS");
        let g = |b: u64| b as f64 / 1e9;
        eprintln!(
            "ram {:.2} | free {:.2} + file-backed {:.2} + purgeable {:.2} = reclaimable {:.2} GB | \
             anonymous {:.2} wired {:.2} compressor {:.2} | own {:.3} | held elsewhere {:.2} GB",
            g(h.ram),
            g(h.free),
            g(h.file_backed),
            g(h.purgeable),
            g(h.reclaimable()),
            g(h.anonymous),
            g(h.wired),
            g(h.compressor),
            g(h.own),
            g(h.held_elsewhere()),
        );
    }
}
