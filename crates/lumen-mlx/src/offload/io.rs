//! Parallel positional reads of expert bytes into slot memory.
//!
//! Measured on this Mac's SSD (random reads, `F_NOCACHE`): about 6.5 GB/s from
//! four threads once reads are 1 MB or larger, and under 2 GB/s through mmap
//! page faults. An expert is a handful of 50 KB to 800 KB ranges, so a miss
//! costs several reads and the pool exists to run them side by side.
//!
//! Two rules come from other projects' failures rather than from theory:
//!
//! * **Never read straight into GPU-visible memory.** Slotstream hit a 42-minute
//!   kernel stall inside a single `pread` aimed at a shared Metal buffer. Each
//!   worker reads into its own private, 16 KiB-aligned scratch buffer and
//!   copies the bytes into the destination afterwards.
//! * **A short read is an error.** `pread` may return fewer bytes than asked; a
//!   loop finishes it, and hitting end-of-file first is reported, never padded.
//!
//! [`IoPool::run`] blocks until the whole batch is done. A worker that fails
//! (or even panics) still counts its batch as finished, so the caller can
//! never be left waiting.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};

/// Alignment of each worker's scratch buffer: the Apple silicon page size.
const SCRATCH_ALIGN: usize = 16 * 1024;

/// Whether reads go through the OS page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IoMode {
    /// `F_NOCACHE` and no read-ahead: the bytes bypass the page cache, so a
    /// model far larger than RAM does not push everything else out of it.
    Direct,
    /// Ordinary buffered reads: repeat reads of the same range can be served
    /// from memory. Which is faster depends on how much RAM is left over.
    Buffered,
    /// Buffered while the machine has memory to spare, Direct when it does not
    /// (see [`choose_direct`]). Measured on Flash-Next at a 6.7 GB pool: +11% with
    /// ample free RAM, -8% on a 24 GB-class Mac under memory pressure.
    Auto,
}

/// Available memory above which `Auto` caches reads in the page cache. Measured on a 36 GB
/// Mac with a 6.7 GB expert pool: ~14 GB available while idle (Buffered wins by 11%), ~4 GB
/// after a 12 GiB balloon took the memory a 24 GB-class Mac would not have (Direct wins by 8%).
const BUFFERED_ABOVE: u64 = 12 << 30;
/// Available memory below which `Auto` goes back to bypassing it. The gap is hysteresis, so a
/// reading that hovers around one value does not flip the mode every sample.
const DIRECT_BELOW: u64 = 8 << 30;
/// How often `Auto` looks at free memory.
const SAMPLE_EVERY: Duration = Duration::from_millis(250);

/// The mode `Auto` should be in, given the one it is in and the free memory now.
pub(crate) fn choose_direct(currently_direct: bool, free_bytes: u64) -> bool {
    if currently_direct {
        free_bytes < BUFFERED_ABOVE
    } else {
        free_bytes < DIRECT_BELOW
    }
}

/// Memory that can serve as page cache without squeezing anything else: free and speculative
/// pages plus file-backed pages, which the OS recycles first.
///
/// Free pages alone are the wrong signal: caching reads fills them, so the very mode this is
/// choosing would drive the number down and switch itself off. File-backed pages do not shrink
/// as the cache grows, and wired or anonymous memory (the expert pool, the weights, another
/// app) is what takes them away.
fn free_memory_bytes() -> Option<u64> {
    fn sysctl_u32(name: &str) -> Option<u64> {
        let name = std::ffi::CString::new(name).ok()?;
        let mut value: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        // SAFETY: `value` and `len` are valid for the call; the name is NUL-terminated.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut u32).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(u64::from(value))
    }
    // SAFETY: a plain query.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    let pages = sysctl_u32("vm.page_free_count")?
        + sysctl_u32("vm.page_speculative_count").unwrap_or(0)
        + sysctl_u32("vm.page_pageable_external_count").unwrap_or(0);
    Some(pages * page)
}

/// The state `IoMode::Auto` keeps.
struct AutoMode {
    direct: AtomicBool,
    last_sample: Mutex<Instant>,
    last_free: AtomicU64,
}

/// One read: `len` bytes at `offset` of file `file`, copied to `dst`.
pub(crate) struct ReadJob {
    pub file: usize,
    pub offset: u64,
    pub len: usize,
    pub dst: *mut u8,
}

// SAFETY: the destination is only written by the one worker that owns the job,
// and `IoPool::run` is `unsafe` precisely so its caller vouches for the pointer.
unsafe impl Send for ReadJob {}

struct Batch {
    jobs: Vec<ReadJob>,
    done: Arc<Done>,
    sent: Instant,
}

/// Where the workers' time goes, summed over every job since the pool opened.
/// Relaxed counters: they are read for a log line, never to decide anything.
#[derive(Default)]
pub(crate) struct IoProfile {
    /// Shards handed to workers, and the time from handing one over until a
    /// worker picked it up (a sleeping worker has to be woken first).
    pub shards: AtomicU64,
    pub wake_ns: AtomicU64,
    /// Reads, bytes, and the time inside `pread` / copying into the slot.
    pub jobs: AtomicU64,
    pub bytes: AtomicU64,
    pub read_ns: AtomicU64,
    pub copy_ns: AtomicU64,
}

impl IoProfile {
    /// One log fragment; averages are per shard (wake) and per read (the rest).
    pub(crate) fn summary(&self) -> String {
        let n = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64;
        let (shards, jobs) = (n(&self.shards).max(1.0), n(&self.jobs).max(1.0));
        format!(
            "io: wake {:.0} us/shard, {:.0} KB/read, pread {:.0} us + copy {:.0} us per read",
            n(&self.wake_ns) / shards / 1e3,
            n(&self.bytes) / jobs / 1e3,
            n(&self.read_ns) / jobs / 1e3,
            n(&self.copy_ns) / jobs / 1e3,
        )
    }
}

/// How long the caller of `run` polls before it goes to sleep on the condvar.
///
/// A sleeping thread is slow to come back: measured on an M-series Mac, the same
/// layer call took 2.7x longer (GPU wait 520 -> 1,430 us, host graph building
/// 130 -> 325 us) when the thread slept 850 us in between, and unchanged when it
/// spun for the same 850 us. A miss costs 1-3 ms of reads, so polling for a few
/// milliseconds keeps the thread hot; anything longer is a slow disk and the
/// core is better given back.
const SPIN_LIMIT: Duration = Duration::from_millis(4);

/// Completion latch for one `run`.
struct Done {
    remaining: AtomicUsize,
    /// Held while checking `remaining` before sleeping, and by the last finisher
    /// while notifying, so a wakeup can never be lost.
    sleep: Mutex<()>,
    cv: Condvar,
    first_error: Mutex<Option<String>>,
}

impl Done {
    fn new(shards: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(shards),
            sleep: Mutex::new(()),
            cv: Condvar::new(),
            first_error: Mutex::new(None),
        }
    }

    fn finish_one(&self, error: Option<String>) {
        if let Some(e) = error {
            let mut slot = self.first_error.lock().unwrap_or_else(|p| p.into_inner());
            slot.get_or_insert(e);
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = self.sleep.lock().unwrap_or_else(|p| p.into_inner());
            self.cv.notify_all();
        }
    }

    /// Poll for `SPIN_LIMIT`, then sleep until every shard has finished.
    fn wait(&self) {
        let give_up = Instant::now() + SPIN_LIMIT;
        while self.remaining.load(Ordering::Acquire) > 0 {
            if Instant::now() >= give_up {
                let mut guard = self.sleep.lock().unwrap_or_else(|p| p.into_inner());
                while self.remaining.load(Ordering::Acquire) > 0 {
                    guard = self.cv.wait(guard).unwrap_or_else(|p| p.into_inner());
                }
                return;
            }
            std::hint::spin_loop();
        }
    }
}

/// Bypass (or use) the page cache for reads on `f`, and turn read-ahead off alongside it.
fn set_direct_on(f: &File, direct: bool) {
    // SAFETY: plain fcntl calls on an fd we own.
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, i32::from(direct));
        libc::fcntl(f.as_raw_fd(), libc::F_RDAHEAD, i32::from(!direct));
    }
}

/// A fixed set of reader threads over a fixed set of files.
pub(crate) struct IoPool {
    profile: Arc<IoProfile>,
    files: Arc<Vec<File>>,
    auto: Option<AutoMode>,
    tx: Option<Sender<Batch>>,
    workers: Vec<JoinHandle<()>>,
    lanes: usize,
}

impl IoPool {
    /// Open `paths` once, shared by `lanes` worker threads.
    pub(crate) fn new(paths: &[PathBuf], lanes: usize, mode: IoMode) -> Result<Self> {
        let lanes = lanes.max(1);
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            let f = File::open(p).with_context(|| format!("io pool: open({})", p.display()))?;
            // `Auto` starts Direct and is moved by the first sample.
            if mode != IoMode::Buffered {
                set_direct_on(&f, true);
            }
            files.push(f);
        }
        let files = Arc::new(files);
        let (tx, rx) = channel::<Batch>();
        let rx: Arc<Mutex<Receiver<Batch>>> = Arc::new(Mutex::new(rx));
        let profile = Arc::new(IoProfile::default());
        let mut workers = Vec::with_capacity(lanes);
        for i in 0..lanes {
            let (files, rx, profile) = (Arc::clone(&files), Arc::clone(&rx), Arc::clone(&profile));
            workers.push(
                std::thread::Builder::new()
                    .name(format!("lumen-expert-io-{i}"))
                    .spawn(move || worker(files, rx, profile))
                    .context("io pool: spawn worker")?,
            );
        }
        let auto = (mode == IoMode::Auto).then(|| AutoMode {
            direct: AtomicBool::new(true),
            last_sample: Mutex::new(Instant::now() - SAMPLE_EVERY),
            last_free: AtomicU64::new(0),
        });
        Ok(Self {
            profile,
            files: Arc::clone(&files),
            auto,
            tx: Some(tx),
            workers,
            lanes,
        })
    }

    pub(crate) fn profile(&self) -> &IoProfile {
        &self.profile
    }

    /// For `Auto`: look at free memory (at most every [`SAMPLE_EVERY`]) and move every file
    /// between Direct and Buffered reads. A no-op for the fixed modes.
    fn retune(&self) {
        let Some(auto) = &self.auto else {
            return;
        };
        {
            let mut last = auto.last_sample.lock().unwrap_or_else(|p| p.into_inner());
            if last.elapsed() < SAMPLE_EVERY {
                return;
            }
            *last = Instant::now();
        }
        let Some(free) = free_memory_bytes() else {
            return;
        };
        auto.last_free.store(free, Ordering::Relaxed);
        let was_direct = auto.direct.load(Ordering::Relaxed);
        let direct = choose_direct(was_direct, free);
        if direct != was_direct {
            for f in self.files.iter() {
                set_direct_on(f, direct);
            }
            auto.direct.store(direct, Ordering::Relaxed);
            eprintln!(
                "[offload] expert reads now {} ({:.1} GB of memory available)",
                if direct {
                    "direct (page cache bypassed)"
                } else {
                    "buffered (page cache used)"
                },
                free as f64 / 1e9
            );
        }
    }

    /// What `Auto` is doing now, with the free memory it last saw; `None` for the fixed modes.
    pub(crate) fn auto_state(&self) -> Option<(bool, u64)> {
        self.auto.as_ref().map(|a| {
            (
                a.direct.load(Ordering::Relaxed),
                a.last_free.load(Ordering::Relaxed),
            )
        })
    }

    /// Run every job and wait for all of them.
    ///
    /// Jobs are spread over the lanes so no lane holds much more than the
    /// others. On error, which jobs completed is unspecified: the caller treats
    /// every destination as unwritten.
    ///
    /// # Safety
    /// Every `dst` must be valid for `len` writable bytes until this returns,
    /// destinations must not overlap each other, and nothing else may read or
    /// write them meanwhile.
    pub(crate) unsafe fn run(&self, jobs: Vec<ReadJob>) -> Result<()> {
        // SAFETY: the caller's contract is `submit`'s, and the ticket is waited on right here.
        unsafe { self.submit(jobs)? }.wait()
    }

    /// Start every job and return at once; [`IoTicket::wait`] collects the result.
    ///
    /// # Safety
    /// As [`Self::run`], but the destinations must stay valid and untouched until the
    /// ticket has been waited on (dropping it waits).
    pub(crate) unsafe fn submit(&self, jobs: Vec<ReadJob>) -> Result<IoTicket> {
        if jobs.is_empty() {
            return Ok(IoTicket { done: None });
        }
        self.retune();
        let shards = shard(jobs, self.lanes);
        let done = Arc::new(Done::new(shards.len()));
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("io pool is shut down"))?;
        for jobs in shards {
            tx.send(Batch {
                jobs,
                done: Arc::clone(&done),
                sent: Instant::now(),
            })
            .map_err(|_| anyhow!("io pool workers are gone"))?;
        }
        Ok(IoTicket { done: Some(done) })
    }
}

/// A batch of reads in flight.
#[must_use = "the reads are only known to be finished after `wait`"]
pub(crate) struct IoTicket {
    done: Option<Arc<Done>>,
}

impl IoTicket {
    /// Block until every read has finished and report the first error.
    pub(crate) fn wait(mut self) -> Result<()> {
        self.collect()
    }

    fn collect(&mut self) -> Result<()> {
        let Some(done) = self.done.take() else {
            return Ok(());
        };
        done.wait();
        match done
            .first_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
        {
            Some(e) => Err(anyhow!(e)),
            None => Ok(()),
        }
    }
}

impl Drop for IoTicket {
    fn drop(&mut self) {
        // The destinations are only valid until the workers are done with them.
        let _ = self.collect();
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        drop(self.tx.take()); // workers see the channel close and exit
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Split into at most `lanes` shards of similar total bytes: biggest job first,
/// each to the lightest shard so far.
fn shard(mut jobs: Vec<ReadJob>, lanes: usize) -> Vec<Vec<ReadJob>> {
    let n = lanes.min(jobs.len()).max(1);
    jobs.sort_by_key(|j| std::cmp::Reverse(j.len));
    let mut shards: Vec<(usize, Vec<ReadJob>)> = (0..n).map(|_| (0, Vec::new())).collect();
    for job in jobs {
        let lightest = shards
            .iter_mut()
            .min_by_key(|(bytes, _)| *bytes)
            .expect("n >= 1");
        lightest.0 += job.len;
        lightest.1.push(job);
    }
    shards.into_iter().map(|(_, v)| v).collect()
}

/// A 16 KiB-aligned byte buffer that grows on demand.
struct Scratch {
    ptr: *mut u8,
    cap: usize,
}

impl Scratch {
    fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    fn get(&mut self, len: usize) -> &mut [u8] {
        if len > self.cap {
            self.free();
            let cap = len.next_multiple_of(SCRATCH_ALIGN);
            let layout = std::alloc::Layout::from_size_align(cap, SCRATCH_ALIGN).expect("layout");
            // SAFETY: non-zero size, valid alignment; checked for null below.
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null(), "scratch allocation of {cap} bytes failed");
            self.ptr = ptr;
            self.cap = cap;
        }
        // SAFETY: `ptr` is valid for `cap >= len` bytes and exclusively ours.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    fn free(&mut self) {
        if !self.ptr.is_null() {
            let layout =
                std::alloc::Layout::from_size_align(self.cap, SCRATCH_ALIGN).expect("layout");
            // SAFETY: allocated by `get` with this exact layout.
            unsafe { std::alloc::dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.free();
    }
}

fn worker(files: Arc<Vec<File>>, rx: Arc<Mutex<Receiver<Batch>>>, profile: Arc<IoProfile>) {
    let mut scratch = Scratch::new();
    loop {
        // Hold the lock only while waiting for the next batch.
        let batch = {
            let guard = rx.lock().unwrap_or_else(|p| p.into_inner());
            match guard.recv() {
                Ok(b) => b,
                Err(_) => return,
            }
        };
        let Batch { jobs, done, sent } = batch;
        profile.shards.fetch_add(1, Ordering::Relaxed);
        profile
            .wake_ns
            .fetch_add(sent.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for job in &jobs {
                read_one(&files, &mut scratch, job, &profile)?;
            }
            Ok::<(), String>(())
        }));
        let error = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e),
            Err(_) => Some("an IO worker panicked".to_string()),
        };
        done.finish_one(error);
    }
}

fn read_one(
    files: &[File],
    scratch: &mut Scratch,
    job: &ReadJob,
    profile: &IoProfile,
) -> Result<(), String> {
    let file = files
        .get(job.file)
        .ok_or_else(|| format!("read job names file {} of {}", job.file, files.len()))?;
    let buf = scratch.get(job.len);
    let started = Instant::now();
    let mut filled = 0usize;
    while filled < job.len {
        match file.read_at(&mut buf[filled..], job.offset + filled as u64) {
            Ok(0) => {
                return Err(format!(
                    "short read: file {} offset {} wanted {} bytes, hit end of file after {filled}",
                    job.file, job.offset, job.len
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => {
                return Err(format!(
                    "read file {} offset {} len {}: {e}",
                    job.file, job.offset, job.len
                ));
            }
        }
    }
    let read_done = Instant::now();
    // SAFETY: the caller of `IoPool::run` guaranteed `dst` is valid for `len`
    // bytes, unaliased and unread until the batch completes; `buf` is our own
    // scratch, so the regions cannot overlap.
    unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), job.dst, job.len) };
    profile.jobs.fetch_add(1, Ordering::Relaxed);
    profile.bytes.fetch_add(job.len as u64, Ordering::Relaxed);
    profile.read_ns.fetch_add(
        read_done.duration_since(started).as_nanos() as u64,
        Ordering::Relaxed,
    );
    profile
        .copy_ns
        .fetch_add(read_done.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A file of `n` pseudo-random bytes, returned with its contents.
    fn file_with(dir: &std::path::Path, name: &str, n: usize, seed: u64) -> (PathBuf, Vec<u8>) {
        let mut s = seed | 1;
        let data: Vec<u8> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 32) as u8
            })
            .collect();
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().write_all(&data).unwrap();
        (p, data)
    }

    fn job(file: usize, offset: u64, dst: &mut [u8]) -> ReadJob {
        ReadJob {
            file,
            offset,
            len: dst.len(),
            dst: dst.as_mut_ptr(),
        }
    }

    #[test]
    fn many_ranges_from_several_files_land_exactly_where_asked() {
        let dir = tempfile::tempdir().unwrap();
        let (p0, d0) = file_with(dir.path(), "a", 3 << 20, 1);
        let (p1, d1) = file_with(dir.path(), "b", 2 << 20, 2);
        let datas = [&d0, &d1];
        for mode in [IoMode::Direct, IoMode::Buffered] {
            for lanes in [1, 3, 8] {
                let pool = IoPool::new(&[p0.clone(), p1.clone()], lanes, mode).unwrap();
                let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ lanes as u64;
                let mut next = || {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng
                };
                let want: Vec<(usize, u64, usize)> = (0..24)
                    .map(|_| {
                        let f = (next() % 2) as usize;
                        let len = 1 + (next() % 200_000) as usize;
                        let off = next() % (datas[f].len() - len) as u64;
                        (f, off, len)
                    })
                    .collect();
                let mut bufs: Vec<Vec<u8>> = want.iter().map(|&(_, _, l)| vec![0u8; l]).collect();
                let jobs: Vec<ReadJob> = want
                    .iter()
                    .zip(bufs.iter_mut())
                    .map(|(&(f, o, _), b)| job(f, o, b))
                    .collect();
                // SAFETY: each destination is its own live Vec, untouched until run returns.
                unsafe { pool.run(jobs) }.expect("run");
                for (&(f, o, l), b) in want.iter().zip(&bufs) {
                    assert_eq!(
                        b.as_slice(),
                        &datas[f][o as usize..o as usize + l],
                        "{mode:?} lanes={lanes}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_pool_is_reusable_across_many_runs_and_with_no_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let (p, d) = file_with(dir.path(), "a", 1 << 20, 7);
        let pool = IoPool::new(&[p], 4, IoMode::Direct).unwrap();
        // SAFETY: no jobs, nothing to vouch for.
        unsafe { pool.run(Vec::new()) }.expect("empty run");
        for i in 0..200usize {
            let mut b = vec![0u8; 4096 + i];
            let off = (i * 1000) as u64;
            // SAFETY: `b` outlives the call.
            unsafe { pool.run(vec![job(0, off, &mut b)]) }.expect("run");
            assert_eq!(
                b.as_slice(),
                &d[off as usize..off as usize + b.len()],
                "run {i}"
            );
        }
    }

    #[test]
    fn reading_past_the_end_of_a_file_is_an_error_not_a_padded_success() {
        let dir = tempfile::tempdir().unwrap();
        let (p, d) = file_with(dir.path(), "a", 10_000, 3);
        let pool = IoPool::new(&[p], 2, IoMode::Direct).unwrap();
        let mut ok = vec![0u8; 100];
        let mut past = vec![0xAAu8; 500];
        let jobs = vec![job(0, 0, &mut ok), job(0, d.len() as u64 - 100, &mut past)];
        // SAFETY: both buffers outlive the call.
        let err = unsafe { pool.run(jobs) }.expect_err("the second job runs off the end");
        assert!(format!("{err:#}").contains("short read"), "got: {err:#}");
        // And the pool is still usable afterwards.
        let mut again = vec![0u8; 64];
        // SAFETY: as above.
        unsafe { pool.run(vec![job(0, 10, &mut again)]) }.expect("pool survives a failed batch");
        assert_eq!(again.as_slice(), &d[10..74]);
    }

    #[test]
    fn a_job_naming_a_missing_file_fails_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let (p, _) = file_with(dir.path(), "a", 1000, 4);
        let pool = IoPool::new(&[p], 2, IoMode::Direct).unwrap();
        let mut b = vec![0u8; 10];
        // SAFETY: `b` outlives the call.
        let err = unsafe { pool.run(vec![job(5, 0, &mut b)]) }.expect_err("bad file index");
        assert!(format!("{err:#}").contains("file 5"), "got: {err:#}");
    }

    #[test]
    fn sharding_balances_bytes_and_keeps_every_job() {
        // The shard test never runs the jobs, so any destination pointer will do.
        let mut dummy = vec![0u8; 1];
        let dst = dummy.as_mut_ptr();
        let mk = move |len| ReadJob {
            file: 0,
            offset: 0,
            len,
            dst,
        };
        let jobs: Vec<ReadJob> = [800, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            .into_iter()
            .map(&mk)
            .collect();
        let total: usize = jobs.iter().map(|j| j.len).sum();
        let shards = shard(jobs, 3);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards.iter().map(Vec::len).sum::<usize>(), 12);
        let loads: Vec<usize> = shards
            .iter()
            .map(|s| s.iter().map(|j| j.len).sum())
            .collect();
        assert_eq!(loads.iter().sum::<usize>(), total);
        assert!(
            loads.iter().max().unwrap() - loads.iter().min().unwrap() <= 800,
            "{loads:?}"
        );
        // Fewer jobs than lanes: one shard per job, none empty.
        let two: Vec<ReadJob> = [50, 60].into_iter().map(&mk).collect();
        let s = shard(two, 8);
        assert_eq!(s.len(), 2);
        assert!(s.iter().all(|x| x.len() == 1));
    }

    #[test]
    fn scratch_is_page_aligned_and_grows() {
        let mut s = Scratch::new();
        let a = s.get(10).as_ptr() as usize;
        assert_eq!(a % SCRATCH_ALIGN, 0);
        let b = s.get(100_000);
        assert_eq!(b.len(), 100_000);
        assert_eq!(b.as_ptr() as usize % SCRATCH_ALIGN, 0);
    }
}

#[cfg(test)]
mod auto_tests {
    use super::*;

    #[test]
    fn auto_goes_buffered_only_with_plenty_free_and_back_only_when_it_runs_low() {
        // Direct stays Direct until memory is clearly spare...
        assert!(choose_direct(true, 0));
        assert!(choose_direct(true, BUFFERED_ABOVE - 1));
        assert!(!choose_direct(true, BUFFERED_ABOVE));
        // ...and Buffered stays Buffered through a dip, returning only below the lower bound.
        assert!(!choose_direct(false, BUFFERED_ABOVE - 1));
        assert!(!choose_direct(false, DIRECT_BELOW));
        assert!(choose_direct(false, DIRECT_BELOW - 1));
    }

    #[test]
    fn this_machines_free_memory_can_be_read() {
        let free = free_memory_bytes().expect("sysctl vm.page_free_count");
        assert!(free > 0 && free < (1 << 40), "{free}");
    }

    #[test]
    fn an_auto_pool_reads_correctly_whichever_mode_it_is_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        let data: Vec<u8> = (0..(1 << 20)).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let pool = IoPool::new(&[path], 4, IoMode::Auto).unwrap();
        for direct in [true, false, true] {
            for f in pool.files.iter() {
                set_direct_on(f, direct);
            }
            let mut out = vec![0u8; 300_000];
            let job = ReadJob {
                file: 0,
                offset: 12_345,
                len: out.len(),
                dst: out.as_mut_ptr(),
            };
            // SAFETY: `out` outlives the call and nothing else touches it.
            unsafe { pool.run(vec![job]) }.unwrap();
            assert_eq!(out[..], data[12_345..12_345 + 300_000], "direct={direct}");
        }
        assert!(pool.auto_state().is_some());
    }
}
