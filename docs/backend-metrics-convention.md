# Backend Metrics Convention

The Tauri desktop app's METRICS card (tok/s, ms/step, requests/min) is fed by
parsing `lumen-server`'s stderr. The parser lives in
[`crates/lumen-app/src/server.rs`](../crates/lumen-app/src/server.rs)
(`MetricsAccumulator` + `parse_tok_per_sec`). It does not call into the
server programmatically — it scans log lines.

That means: **a new model backend that doesn't emit a matching line will
make the METRICS card stay blank for that model.** This was the actual
bug fixed for Gemma 4 in commit `fe0b736 + <this commit>` — the Gemma 4
path never emitted a `tok/s` line, so the EMA never updated.

## The contract

### 1. Format

Emit **exactly one** line at the end of each chat / completion request:

```
[<backend>] <kind> done: <N> tokens in <T_ms>ms (<R> tok/s)
```

| Field | Meaning | Example |
|---|---|---|
| `<backend>` | short tag identifying the backend | `gemma4`, `mlx`, `mlx-spec`, `qwen35 batched` |
| `<kind>` | usually `chat` for `/v1/chat/completions`, `completion` for `/v1/completions`, or `seq N` in the multi-sequence batched engine | `chat`, `completion`, `seq 7` |
| `<N>` | decode tokens generated (NOT including prompt) | `128` |
| `<T_ms>` | decode wall time in ms, integer (`.0` precision) | `2143` |
| `<R>` | decode rate, one decimal (`.1` precision) | `59.7` |

The parser requires **two literal tokens** anywhere in the line:

1. `done:` — distinguishes decode finalization from prefill / per-step
   diagnostics that happen to also carry a `tok/s` reading.
2. ` tok/s)` — space + rate + close paren. The parser walks backward from
   the close paren to collect the number.

### 2. Stream

Write to **stderr** only. Use `eprintln!`. The supervisor's `pipe_to_event`
only feeds stderr lines into `MetricsAccumulator::observe`
([server.rs:399-406](../crates/lumen-app/src/server.rs#L399-L406)).

Do **not** use `tracing::info!` or `println!` for this line — both would
miss the parser. (You may continue to use them for everything else.)

### 3. Frequency

**Once per request, at decode end.** Not per token, not per step, not on
prefill.

If the request was canceled (client disconnect, max_new_tokens hit, EOS),
emit on the exit path too — pick whatever timing you have. A partial-decode
tok/s is more useful than a missing one.

### 4. Performance

`eprintln!` is one syscall (~1-10 µs). Compared to a chat completion
(hundreds of ms to multiple seconds), this is < 0.001% overhead. The
parser does a substring scan + one `f64::parse` per stderr line, also
microseconds.

But: do not emit per-token or per-step. The mutex held in
`pipe_to_event` while observing is brief, but per-token traffic would
serialize the stderr pump against everything else.

## Canonical examples

```rust
// crates/lumen-mlx/src/lib.rs — Qwen35 chat_streaming
let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
let n_gen = generated.len();
eprintln!(
    "[mlx] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
    n_gen as f64 / (decode_ms / 1000.0)
);

// crates/lumen-mlx/src/gemma4_backend.rs — Gemma 4 chat (stats-based)
eprintln!(
    "[gemma4] chat done: {} tokens in {:.0}ms ({:.1} tok/s)",
    stats.decode_steps, stats.decode_ms, stats.decode_tok_per_sec
);

// crates/lumen-mlx/src/gemma4_backend.rs — Gemma 4 chat_streaming (manual)
let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
let tok_per_sec = if decode_ms > 0.0 && count > 0 {
    count as f64 / (decode_ms / 1000.0)
} else {
    0.0
};
eprintln!(
    "[gemma4] chat done: {count} tokens in {decode_ms:.0}ms ({tok_per_sec:.1} tok/s)"
);
```

## Anti-examples (will be ignored)

| Line | Why it doesn't match |
|---|---|
| `[mlx] seq 7 prefill: 4096 tokens in 1500ms (2730.7 tok/s) -> tok=42` | no `done:` |
| `[mlx] seq 7 EOS at step 42 (28.3 tok/s)` | no `done:` (intermediate event) |
| `[batched engine] step: N=4 latency=42.1ms aggregate=15.3 tok/s` | no close-paren after rate |
| `[xxx] done: 42 tokens (27.5 tokens/sec)` | wrong unit suffix (`tokens/sec` vs `tok/s`) |
| `[xxx] chat finished: 42 tokens (27.5 tok/s)` | no `done:` keyword |

## Flow end-to-end

1. Backend emits `[xxx] ... done: ... (R tok/s)` to stderr.
2. [`pipe_to_event`](../crates/lumen-app/src/server.rs) reads the line.
3. [`MetricsAccumulator::observe`](../crates/lumen-app/src/server.rs) calls
   `parse_tok_per_sec`.
4. If matched and in `(0.1..=1000.0)`, the value enters an EMA with α=0.3.
   `ms_per_step` is derived as `1000.0 / rate_ema`.
5. `request_times: VecDeque<Instant>` gets a stamp; entries older than 60s
   are evicted. Length = `requests_per_min`.
6. Frontend polls `server_metrics` every 2 s, renders the values.

## Optional second format (SSE stream-timing)

If `LUMEN_STREAM_TIMING=1`, the streaming code in
[`crates/lumen-server/src/routes/chat.rs`](../crates/lumen-server/src/routes/chat.rs)
emits an additional line:

```
[stream-timing] sse: n_deltas=42 ... steady_rate_recv=22.78tok/s ...
```

This is also parsed (the `steady_rate_recv=` branch) and feeds the same
EMA. You don't need to emit this — it's the SSE adapter's job, not the
model backend's. Mentioning it here so you know that two values per
request might enter the EMA when this flag is on (one from decode-done,
one from the SSE wire steady rate).

## Adding tests

Add a positive case in `parse_tests` in
[`crates/lumen-app/src/server.rs`](../crates/lumen-app/src/server.rs):

```rust
#[test]
fn parses_<your_backend>_done() {
    let line = "[<backend>] chat done: 42 tokens in 1530ms (27.5 tok/s)";
    assert!((parse_tok_per_sec(line).unwrap() - 27.5).abs() < 1e-6);
}
```

Run with `cargo test -p lumen-app --lib parse_tests`. If your line shape
doesn't fit the convention exactly, the test will fail — fix the line
shape, not the parser.

## See also

- [`crates/lumen-app/src/server.rs`](../crates/lumen-app/src/server.rs) —
  parser + accumulator (read it; it's ~150 lines)
- [`crates/lumen-mlx/src/lib.rs`](../crates/lumen-mlx/src/lib.rs) — Qwen35
  emission sites
- [`crates/lumen-mlx/src/gemma4_backend.rs`](../crates/lumen-mlx/src/gemma4_backend.rs)
  — Gemma 4 emission sites (4 paths: `generate`, `chat`,
  `chat_with_prefix_cache`, `chat_streaming`)
- [`crates/lumen-server/src/engine.rs`](../crates/lumen-server/src/engine.rs)
  — batched-engine emission sites (`[batched engine]`, `[qwen35 batched]`)
