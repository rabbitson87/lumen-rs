//! Routing trace of the expert pool: what every layer call asked for, and what the real
//! engine did with it. It exists to replay a workload against other eviction policies offline
//! (a trace-driven simulator); it never feeds back into the engine.
//!
//! `LUMEN_MOE_TRACE=<path>` makes [`super::MoeOffload`] write one binary file at that path
//! (created or truncated when the offload opens). When the variable is unset or `0` there is
//! no recorder at all: [`super::MoeOffload`] holds an `Option`, so a layer call pays one
//! `is_some` check and allocates nothing.
//!
//! Everything is little endian.
//!
//! ```text
//! header   "LMOETR01"  u32 version(=2)  u64 run_id (unix ns at open)
//!          u64 budget_bytes  u32 n_slots  u32 regions  u32 width  u32 top_k
//!          u64 expert_bytes  u32 bulk_min_percent  u32 decode_rows
//!          u32 n_layers  u16 layer[n_layers]            (MoE layers, MTP head included)
//! record   u8 tag(=1)  u8 flags  u16 layer  u32 request  u32 pass  u32 rows  u16 n
//!          u16 expert[n]            distinct experts of the call, first-seen order
//!          u8 hit[ceil(n/8)]        bit i set: expert[i] was resident when the call looked
//!          u64 t_enter  u64 t_eval  u64 t_done      ns since the trace opened (see below)
//!          [counts]                 only with bit5: u16 total[n]  u16 tail[n]
//! flags    bit0 staged   the call took the staging path (pool untouched; hit = resident then)
//!          bit1 hinted   a layer-major prefill had switched every call to staging
//!          bit2 prefill  hinted, or more than `decode_rows` rows
//!          bit3 pass     first call of a new pass (a decode token, or a prefill group)
//!          bit4 request  first call of a new request
//!          bit5 counts   the call has more than `decode_rows` rows, so per-expert token counts follow
//! ```
//!
//! `total[i]` is how many (token, slot) routings of this call chose `expert[i]` (a token routes to
//! `top_k` distinct experts, so the counts sum to `rows * top_k`); `tail[i]` counts only the last
//! `TAIL_ROWS` tokens of the call. Timestamps: `t_enter` is when `route` was entered (the GPU is
//! still working on everything before this layer), `t_eval` when the routing had been read back
//! (the GPU is now idle), `t_done` when `route` returned (experts resident or staged). So the SSD
//! is busy for about `t_done - t_eval` on a call that read, and `t_enter - previous t_done` is the
//! host building the next layer's graph.
//!
//! `request` counts the switches from decoding to prefilling; `pass` counts decode tokens or
//! prefill groups inside it, restarting at 0 when the kind changes. Both are derived from the
//! call sequence (a layer number lower than the previous call's starts a new pass), so a
//! request that never decodes merges into the next one's prefill. A prompt of `decode_rows` or
//! fewer tokens that is not part of a layer-major group would look like decoding.
//!
//! On the acquire path (`staged` clear) `hit` is exactly the engine's own hit flag, and the
//! engine's hit/miss counters are the sums of these flags. On the staging path the engine counts
//! nothing; `hit` is residency in the pool after the staging blocks were borrowed from it.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};

lumen_flags::flag! {
    /// Write a binary trace of the expert pool's layer calls (expert ids in request order, the
    /// engine's hit flags, request/pass markers, per-expert token counts for prefill calls and
    /// per-call timestamps) to the path in this variable, for replaying a workload against other
    /// eviction policies offline. Any value but `0` is the output path. Never changes the output
    /// or what is resident; costs one `Option` check per layer call when off.
    pub(crate) moe_trace {
        env: "LUMEN_MOE_TRACE",
        default: false,
        kind: Diagnostic,
    }
}

/// The variable's name, for reading its value after the registry flag said it is set.
const TRACE_ENV: &str = "LUMEN_MOE_TRACE";

const MAGIC: &[u8; 8] = b"LMOETR01";
const VERSION: u32 = 2;
const TAG_CALL: u8 = 1;
/// More rows than this is a prefill chunk; fewer is a decode step or a speculative verify.
pub(crate) const DECODE_ROWS: u32 = 4;
/// The tail counts cover this many last tokens of a call.
pub(crate) const TAIL_ROWS: u32 = 512;
/// Flush to the file once this much is buffered, and at the last layer of every pass.
const FLUSH_BYTES: usize = 1 << 16;

const F_STAGED: u8 = 1;
const F_HINTED: u8 = 1 << 1;
const F_PREFILL: u8 = 1 << 2;
const F_PASS: u8 = 1 << 3;
const F_REQUEST: u8 = 1 << 4;
const F_COUNTS: u8 = 1 << 5;

/// What the engine looked like when the trace opened.
pub(crate) struct Header {
    pub budget_bytes: u64,
    pub n_slots: usize,
    pub regions: usize,
    pub width: usize,
    pub top_k: usize,
    pub expert_bytes: u64,
    pub bulk_min_percent: usize,
    pub layers: Vec<u16>,
}

/// One layer call, as the engine saw it.
pub(crate) struct Call<'a> {
    pub layer: u16,
    /// Tokens routed in this call.
    pub rows: u32,
    pub staged: bool,
    pub hinted: bool,
    /// Distinct experts, first-seen order.
    pub ids: &'a [u16],
    /// Same length as `ids`.
    pub hits: &'a [bool],
    /// `(total, tail)` token counts per expert, parallel to `ids`; for calls with more than
    /// [`DECODE_ROWS`] rows.
    pub counts: Option<(&'a [u16], &'a [u16])>,
    pub entered: Instant,
    pub evaluated: Instant,
    pub done: Instant,
}

pub(crate) struct RouteTrace {
    state: Mutex<State>,
}

struct State {
    epoch: Instant,
    out: File,
    buf: Vec<u8>,
    /// Highest trunk layer: its call ends a pass.
    last_layer: u16,
    prev: Option<(u16, bool)>,
    request: u32,
    pass: u32,
    broken: bool,
}

impl RouteTrace {
    /// The recorder when `LUMEN_MOE_TRACE` is set, else `None`. `header` runs only when it is.
    pub(crate) fn from_env(header: impl FnOnce() -> Header) -> Result<Option<Self>> {
        if !moe_trace::get() {
            return Ok(None);
        }
        let path = std::env::var(TRACE_ENV).unwrap_or_default();
        Self::create(PathBuf::from(path), &header()).map(Some)
    }

    fn create(path: PathBuf, header: &Header) -> Result<Self> {
        let mut out = File::create(&path)
            .with_context(|| format!("LUMEN_MOE_TRACE: create {}", path.display()))?;
        let mut head = Vec::with_capacity(96 + 2 * header.layers.len());
        head.extend_from_slice(MAGIC);
        head.extend_from_slice(&VERSION.to_le_bytes());
        let run_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        head.extend_from_slice(&run_id.to_le_bytes());
        head.extend_from_slice(&header.budget_bytes.to_le_bytes());
        for v in [header.n_slots, header.regions, header.width, header.top_k] {
            head.extend_from_slice(&(v as u32).to_le_bytes());
        }
        head.extend_from_slice(&header.expert_bytes.to_le_bytes());
        head.extend_from_slice(&(header.bulk_min_percent as u32).to_le_bytes());
        head.extend_from_slice(&DECODE_ROWS.to_le_bytes());
        head.extend_from_slice(&(header.layers.len() as u32).to_le_bytes());
        for l in &header.layers {
            head.extend_from_slice(&l.to_le_bytes());
        }
        out.write_all(&head)
            .with_context(|| format!("LUMEN_MOE_TRACE: write {}", path.display()))?;
        eprintln!("[offload] routing trace -> {}", path.display());
        let last_layer = header
            .layers
            .iter()
            .copied()
            .filter(|&l| l != super::MTP_LAYER)
            .max()
            .unwrap_or(0);
        Ok(Self {
            state: Mutex::new(State {
                epoch: Instant::now(),
                out,
                buf: Vec::with_capacity(2 * FLUSH_BYTES),
                last_layer,
                prev: None,
                request: 0,
                pass: 0,
                broken: false,
            }),
        })
    }

    /// Append one call. A write error is reported once and silences the recorder: the trace
    /// must never be able to fail a request.
    pub(crate) fn record(&self, call: &Call<'_>) {
        debug_assert_eq!(call.ids.len(), call.hits.len());
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.broken {
            return;
        }
        let prefill = call.hinted || call.rows > DECODE_ROWS;
        let (new_request, new_pass) = match st.prev {
            None => (true, true),
            Some((prev_layer, prev_prefill)) => {
                let new_request = prefill && !prev_prefill;
                if new_request {
                    st.request += 1;
                    st.pass = 0;
                } else if prefill != prev_prefill {
                    // Prefill -> decode inside a request: the first token is token 0.
                    st.pass = 0;
                } else if call.layer < prev_layer {
                    st.pass += 1;
                }
                (
                    new_request,
                    new_request || prefill != prev_prefill || call.layer < prev_layer,
                )
            }
        };
        st.prev = Some((call.layer, prefill));

        let flags = [
            (call.staged, F_STAGED),
            (call.hinted, F_HINTED),
            (prefill, F_PREFILL),
            (new_pass, F_PASS),
            (new_request, F_REQUEST),
            (call.counts.is_some(), F_COUNTS),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .fold(0u8, |acc, (_, bit)| acc | bit);
        let ns = |t: Instant| t.saturating_duration_since(st.epoch).as_nanos() as u64;
        let times = [ns(call.entered), ns(call.evaluated), ns(call.done)];
        let (request, pass) = (st.request, st.pass);
        let buf = &mut st.buf;
        buf.push(TAG_CALL);
        buf.push(flags);
        buf.extend_from_slice(&call.layer.to_le_bytes());
        buf.extend_from_slice(&request.to_le_bytes());
        buf.extend_from_slice(&pass.to_le_bytes());
        buf.extend_from_slice(&call.rows.to_le_bytes());
        buf.extend_from_slice(&(call.ids.len() as u16).to_le_bytes());
        for e in call.ids {
            buf.extend_from_slice(&e.to_le_bytes());
        }
        let mut bits = vec![0u8; call.ids.len().div_ceil(8)];
        for (i, &hit) in call.hits.iter().enumerate() {
            bits[i / 8] |= u8::from(hit) << (i % 8);
        }
        buf.extend_from_slice(&bits);
        for t in times {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        if let Some((total, tail)) = call.counts {
            debug_assert!(total.len() == call.ids.len() && tail.len() == call.ids.len());
            for c in total.iter().chain(tail) {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }

        if st.buf.len() >= FLUSH_BYTES || call.layer == st.last_layer {
            let State {
                out, buf, broken, ..
            } = &mut *st;
            if let Err(e) = out.write_all(buf) {
                eprintln!("[offload] routing trace: write failed, tracing stops: {e}");
                *broken = true;
            }
            buf.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            budget_bytes: 12_000_000_000,
            n_slots: 4340,
            regions: 2,
            width: 512,
            top_k: 10,
            expert_bytes: 2_764_800,
            bulk_min_percent: 60,
            layers: vec![0, 1, 2],
        }
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    /// Request/pass markers follow the call sequence: a prefill group (two chunks per layer),
    /// then two decode tokens, then a second request's prefill. Records carry timestamps, and
    /// the calls with more than `decode_rows` rows carry per-expert token counts.
    #[test]
    fn markers_and_layout_follow_the_call_sequence() {
        let dir = std::env::temp_dir().join(format!("lumen-moe-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.bin");
        let t = RouteTrace::create(path.clone(), &header()).unwrap();
        let ids = [7u16, 3, 9];
        let hits = [true, false, true];
        let total = [1000u16, 20, 5];
        let tail = [500u16, 3, 0];
        let now = Instant::now();
        let call = |layer, rows, staged, hinted| Call {
            layer,
            rows,
            staged,
            hinted,
            ids: &ids,
            hits: &hits,
            counts: (rows > DECODE_ROWS).then_some((&total[..], &tail[..])),
            entered: now,
            evaluated: now,
            done: now,
        };
        // Prefill group 0: layer 0 twice (two chunks), layer 1 twice, layer 2 twice.
        for layer in [0, 0, 1, 1, 2, 2] {
            t.record(&call(layer, 1024, true, true));
        }
        // Decode token 0 and 1.
        for layer in [0, 1, 2, 0, 1, 2] {
            t.record(&call(layer, 1, false, false));
        }
        // A new request: a single-chunk prefill on the acquire path.
        for layer in [0, 1, 2] {
            t.record(&call(layer, 30, false, false));
        }
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(u32_at(&bytes, 8), 2, "version");
        let head_len = 8 + 4 + 8 + 8 + 16 + 8 + 4 + 4 + 4 + 2 * 3;
        assert_eq!(u32_at(&bytes, head_len - 6 - 4), 3, "layer count");
        // A record: 18 bytes of fixed fields, 3 ids, 1 hit byte, 3 timestamps, and for the calls
        // with counts 3 + 3 u16.
        let plain = 18 + 6 + 1 + 24;
        let counted = plain + 12;
        let mut at = head_len;
        let mut marks = Vec::new();
        while at < bytes.len() {
            let flags = bytes[at + 1];
            let size = if flags & F_COUNTS != 0 {
                counted
            } else {
                plain
            };
            marks.push((
                u32_at(&bytes, at + 4),
                u32_at(&bytes, at + 8),
                flags,
                at,
                size,
            ));
            at += size;
        }
        assert_eq!(at, bytes.len());
        assert_eq!(marks.len(), 15);
        let (req, pass): (Vec<u32>, Vec<u32>) = marks.iter().map(|m| (m.0, m.1)).unzip();
        assert_eq!(req, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(pass, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]);
        let f = |i: usize| marks[i].2;
        assert_eq!(f(0) & (F_PASS | F_REQUEST), F_PASS | F_REQUEST);
        assert_eq!(f(1) & (F_PASS | F_REQUEST), 0, "second chunk, same pass");
        assert_eq!(
            f(2) & (F_PASS | F_REQUEST),
            0,
            "layer 1 continues the group"
        );
        assert_eq!(f(6) & F_PASS, F_PASS, "decode starts a pass");
        assert_eq!(f(6) & (F_PREFILL | F_COUNTS), 0);
        assert_eq!(f(9) & F_PASS, F_PASS, "the layer wrap is the next token");
        assert_eq!(f(12) & F_REQUEST, F_REQUEST);
        assert_eq!(f(12) & F_PREFILL, F_PREFILL, "30 rows is a prefill");
        assert_eq!(f(0) & (F_STAGED | F_COUNTS), F_STAGED | F_COUNTS);
        // The ids and the hit bits (7 hit, 3 miss, 9 hit = 0b101).
        let first = marks[0].3;
        assert_eq!(
            u16::from_le_bytes([bytes[first + 18], bytes[first + 19]]),
            7
        );
        assert_eq!(bytes[first + 24], 0b101);
        // Counts follow the timestamps: totals, then tails.
        let counts = first + 18 + 6 + 1 + 24;
        let at_u16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        assert_eq!(
            [at_u16(counts), at_u16(counts + 2), at_u16(counts + 4)],
            total
        );
        assert_eq!(
            [at_u16(counts + 6), at_u16(counts + 8), at_u16(counts + 10)],
            tail
        );
    }
}
