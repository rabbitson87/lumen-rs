# Qwen3.8-Flash-Next on a Mac with 16–36 GB

Flash-Next (`pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, 125B parameters, 6B active) is a
mixture-of-experts model whose routed experts are 68 GB of its 97 GB. lumen keeps the rest (about 3.6 GB
of attention, DeltaNet and shared weights) in memory and serves the routed experts from the SSD through
a pool of memory slots, so the model runs on a Mac whose RAM is a fraction of its size. The output is
the model's own: an expert's bytes are the checkpoint's bytes, read from disk instead of memory.

This page lists the options. Every one of them changes speed or memory, none changes what the model
computes, except speculative decoding, which is marked.

## What decides the speed

- **Decode** is limited by how many experts a token needs that are not in the pool. A bigger pool
  means fewer reads (hit rate 0.66 at a 6.7 GB pool, 0.87 at 19.8 GB on a 36 GB Mac, over the first 300 tokens).
- **Prefill** reads each layer's experts once per prompt (up to 8,192 tokens at a time), so it is
  limited by the GPU, not the SSD: about 230 tokens/s on an 18K-token prompt at any pool size.
- A turn with a few hundred new tokens still needs nearly every expert of every layer, so its first
  token costs one pass over the SSD (6–13 s) however short the turn is.

## How the memory is planned

The GPU's working set (the part of RAM the OS lets one process keep resident, about 75% of it,
times 0.88 for the headroom other apps need; `LUMEN_MEMORY_LIMIT_GB` lowers it) is split into the
3.6 GB of resident weights, a reserve for prefilling, and the expert pool, which gets the rest. The
reserve is 0.6 GB of buffers, about 1.1 MB per token of the prefill chunk, the KV cache of the
longest prompt the plan allows (60 KB per token), and half the attention scores. So the longest prompt
and the pool trade against each other: `LUMEN_MAX_PROMPT_TOKENS` (the server's prompt cap, which the
plan also reads; `LUMEN_MAX_CTX` can only lower it) is 16,384 by default, and 65,536 takes about 3 GB out of the pool. The server rejects a
longer prompt with the usual "prompt too large" error instead of running out of memory halfway. A
chunk attends to a long context in blocks of query rows so its scores never exceed 2 GB.

| GPU working set | Machine | Chunk | Longest prompt | Pool (auto) | Decode |
|---|---|---|---|---|---|
| 26.5 GB | 36 GB Mac | 1,024 | 16,384 | 19.4 GB | 12.9 tok/s |
| 14 GB (emulated) | 24 GB Mac | 1,024 | 16,384 | 6.85 GB | 7.2 tok/s |
| 9.7 GB (emulated; a real 16 GB Mac reports ~10 GB) | 16 GB Mac | 512 | 8,192 | 4.2 GB | 6.1 tok/s |

Decode figures are with the expert pack, the default eviction and direct reads, on this machine's fast
SSD (6.6 GB/s), at the nearest explicit pool (19.8, 6.7 and 4.2 GB; 300 tokens of prose). The two
smaller rows' plans were made on a 36 GB M3 Max with `LUMEN_MEMORY_LIMIT_GB` set to 13 and 9, so they show
what the plan does, not how a real 16 or 24 GB Mac's own memory pressure behaves. On a 16 GB Mac (under 11 GB of working set) the
pool has room for only one staging region, so prefill cannot read the next layer while computing the
current one; a 5.8K-token prompt prefilled at 209 tok/s there, with a peak resident size of 7.2 GB.

### What else is in memory

The reserve is sized for one prompt, and the server keeps more than one. A chat session holds its KV
twice (the session and its rollback point: 1.9 GB for a 15K-token prompt, plus 0.12 GB of DeltaNet
state per copy), and a request with tools keeps a snapshot of its prompt and of its system prompt's
head (0.4-0.8 GB each at 6K-12K tokens), without a bound. On the emulated 16 and 24 GB Macs a second
long conversation took the live memory past `LUMEN_MEMORY_LIMIT_GB` (14.2 GB against 14.0 for the
24 GB plan), and then MLX stops recycling its buffer cache: it frees the cache at every allocation, so
each of the thousands of buffers a decode step makes is a new Metal buffer. From the second long prompt
on, decode ran at half speed and prefill 15-40% slower (2.7 times, in the twelve-request run below). Nothing leaked: the live memory came back to
10.2 GB when the sessions were dropped, and with one session at a time, or the same plan under a 20 GB
limit, the second prompt decoded at 8.2-8.4 tok/s.

Two different 15K-16K-token prompts in a row (24 GB class) and 7.5K-8K (16 GB class), the text
identical in every run, N=4-5 per row:

| Plan | Decode, first → second prompt | Second prompt's first token |
|---|---|---|
| 24 GB class, kept without a bound | 7.1-7.4 → 4.0-4.1 tok/s | 65-80 s |
| 24 GB class, dropped to fit | 6.9-7.2 → 8.2-8.3 tok/s | 57-67 s |
| 16 GB class, kept without a bound | 7.6-8.1 → 3.8 tok/s | 44 s |
| 16 GB class, dropped to fit | 7.7-7.8 → 7.2-7.3 tok/s | 37 s |

Twelve mixed requests (chat, tool rounds, long prompts), 24 GB class: the long prompts after the first
decoded at 3.5-3.9 tok/s and prefilled 11.5K tokens in 54-99 s; now 6.5-7.7 tok/s and 37-40 s, the
same as with memory to spare (6.1-7.0 tok/s, 36-50 s). The 16 GB class went from 3.6-3.9 to 6.7-7.6
tok/s (control with memory to spare: 6.5-7.6).

So before it feeds a prompt the server adds what the feed will leave in memory (its KV for every copy
kept, and a sequence state) and half a GB for MLX's cache to what is live, and if that is past the
plan it drops, until it fits: the heads of other system prompts (and if that is not enough this
prompt's own head snapshot is not taken; it is still prefilled in the same two stages, so its text
does not change), the least recently used other sessions, then the least recently used prefix
snapshots. The conversation being continued, and the snapshot being forked, are never dropped, and a
feed that leaves under 0.5 GB (a title request, a short turn) drops nothing unless it would pass the
limit by itself. Half a GB is what the cache needs: with two 16K-token conversations kept, 0.18 to
0.64 GB of cache room decoded at 8.4-9.3 tok/s, none at 4.3. The price is that a conversation dropped
this way is prefilled again when it comes back; with more memory, or a shorter
`LUMEN_MAX_PROMPT_TOKENS`, the server keeps more. `LUMEN_MLX_MEMORY_EVICT=0` switches it off, and
`LUMEN_QWEN4_MEM_LOG=1` logs MLX's active, cache and peak memory and what each sequence holds after
every prefill and every 32 decode steps. The plan above (pool sizes, reserve) is unchanged.

## Which expert leaves the pool

When a layer needs an expert that is not in the pool, another one has to give up its slot. The default,
`bias`, keeps a decayed hit count per slot and learns, per layer, which expert tends to follow which at the
next token ("expert 17 at token t was followed by expert 203 at token t + 1"). It evicts, from 64 slots
sampled at random, the one with the lowest count plus predicted-next score. It costs about 2 µs per layer
call and 25 MB. `gclock` is plain CLOCK with a 2-bit counter and `clock` is plain CLOCK.

Chosen by replaying routing traces of 18 real sessions (prose, code, Korean, tool-calling agents,
multi-turn chat, an 8K-token document, reasoning) through a model of the pool that reproduces the
engine's hit/miss decisions exactly (0 differences in 15 million), then checked on the real server:

| Pool | Hit rate, `clock` → `bias` (3 sessions) | Misses | Decode speed, 300 tokens, N=9, `bias` vs `clock` (default I/O mode) |
|---|---|---|---|
| 4 GB | 52.8% → 57.6% | −10% | +3.9% (3.7 σ) |
| 6.7 GB | 64.2% → 68.6% | −12% | +11.1% (11.7 σ) |
| 19 GB | 88.2% → 89.9% | −14% | +3.4% (not significant) |

At 6.7 GB the multi-turn session gained 13–18% per turn. The first token also comes sooner (−5% to −10% at
4 and 6.7 GB). The text is identical for every policy and every pool size. FIFO, round-robin and random
evict 4–7 points worse than `clock` (a token re-asks 34% of its experts at the very next token, so
recency matters). A plain frequency count (LFU with aging) was 16 points worse at the smallest pool and
only 1.6 points better than `clock` at 19.8 GB. Admitting the prompt's experts to the pool after a prefill, bypassing the
pool for big calls and a shipped popularity table gained nothing worth their complexity. At 19 GB `bias` and
`gclock` gain about the same.

**Exactness.** The experts' output must not depend on where in the pool they sit. MLX's sorted `gather_qmm`
changes kernel when a call has at least 4 rows per expert of the tensor it gathers over, and the pool path
gathers over the whole pool, so a call of 205 or more rows used to round differently at every pool size. Such
calls (long prompt chunks) now take the staging path, which gathers over one layer exactly as the resident
model does; this can change the text of a long prompt compared with earlier builds, and it makes the output
the same for every pool size and policy. The check is `cargo xtask offload-check` (it runs the tests in
`offload/exactness_tests.rs`); run it after every MLX upgrade, since the threshold is an MLX internal.

## Estimating the speed on your Mac

A token costs a fixed part plus the SSD reads for the experts that are not in the pool:

- **Fixed:** the GPU streams about 4.9 GB per token (the 1.3 GB of experts it uses plus the 3.6 GB of
  resident weights), so about `4.9 GB ÷ memory bandwidth + 24 ms` (the 24 ms is dispatch and sync; it is
  the least certain constant, ±10 ms, and the predictions barely move with it).
- **Reads:** the experts missing from the pool, `(1 − hit rate) × 1.3 GB`, at the SSD's random read speed
  times 0.63 when reading the shipped shards or 0.72 with the expert pack. The hit rate rises with the
  pool (default eviction, first 300 tokens; longer sessions warm up and do better): 0.48 at 2.8 GB,
  0.55 at 4.2 GB, 0.66 at 6.7 GB, 0.75 at 10 GB, 0.83 at 15 GB, 0.87 at 19.8 GB.

The app measures the two inputs (Tuning → Flash-Next → *Measure this Mac*: the GPU's memory bandwidth in
about a second with `lumen-server --measure-memory-bandwidth`, and the SSD with the Doctor's read probe)
and then estimates the speed for the pool you chose. Until both are measured the card shows only the
table below as a reference, never as a promise. On the Mac it was calibrated on (an M3 Max with 36 GB:
281 GB/s memory, 6.6 GB/s SSD, direct reads) the formula reproduces the twelve measured speeds within 4%;
on other Macs it is an estimate, not a measurement, and the hit-rate curve is the part it cannot know.
Measure with the server stopped and nothing else busy, since a decode or another GPU job shares the GPU
and the SSD and can make the reading come out lower (three runs on an idle Mac here read 281–282 GB/s).
Reference speeds (tokens/s, expert pool set explicitly, direct reads, 300 tokens of prose):

| Pool | Shards | Pack |
|---|---|---|
| 2.8 GB (the floor) | 4.7 | 5.4 |
| 4.2 GB | 5.6 | 6.1 |
| 6.7 GB | 6.7 | 7.2 |
| 10 GB | 8.1 | 9.0 |
| 15 GB | 10.4 | 11.2 |
| 19.8 GB | 12.0 | 13.0 |

With `LUMEN_MOE_IO_MODE=auto` on a Mac that has more than 12 GB to spare, the OS page cache holds
experts too and can make a small pool run faster than these numbers (a 4 GB pool even outran 6.7 GB
there); a 16 or 24 GB Mac has little to spare, which is why the reference uses direct reads.

## Options

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_MOE_EXPERT_CACHE_GB` | `auto` | Size of the expert pool: `auto`, or a size such as `4g`, `8GB`, `12.5g`, `512m` (a plain number is GB). `auto` takes what the GPU's working set leaves after the 3.6 GB of resident weights and the prefill reserve. A size that does not fit beside them (at the longest prompt the plan allows) is lowered to what does, and the log says so; a shorter `LUMEN_MAX_PROMPT_TOKENS` makes room. Measured on a 36 GB M3 Max with the pack and direct reads, 300 tokens: 4.2 GB 6.1 tok/s, 6.7 GB 7.2, 10 GB 9.0, 15 GB 11.2, 19.8 GB 13.0. |
| `LUMEN_QWEN35_PREFILL_CHUNK` | 1024 (512 under 11 GB) | Tokens per prefill chunk. Prefill speed does not depend on it; its activations come out of the pool. |
| `LUMEN_MAX_PROMPT_TOKENS` | 16384 (8192 under 11 GB) | Longest prompt; the KV cache for it is planned out of the pool. |
| `LUMEN_MEMORY_LIMIT_GB` | 32 | Cap on MLX's memory in GiB; the pool is planned within it. |
| `LUMEN_MOE_EVICTION` | `bias` | Which expert leaves the pool when it is full: `bias` (learned, see below), `gclock` or `clock`. Never changes the output. |
| `LUMEN_MOE_IO_MODE` | `auto` | `direct` bypasses the page cache for expert reads, `buffered` uses it, `auto` uses it while the machine has more than 12 GB of memory to spare. |
| `LUMEN_MOE_IO_LANES` | 32 | Reader threads (1–64). |
| `LUMEN_MOE_PACK` | `auto` | The contiguous expert pack: `auto` uses `experts.lpack` when it is next to the weights, `off` never does, `build` makes it first. See below. |
| `LUMEN_QWEN4_MTP` | `off` | The draft head for speculative decoding: `off`, `auto` (use it when installed, with a speed governor) or `on` (the same, but refuses to start without the head). See below. |
| `LUMEN_QWEN4_MTP_FLOOR` | 0.4 | Share of drafts a conversation must keep getting accepted to keep speculating. |

## Speculative decoding (the draft head)

The model ships a small extra layer (the MTP head) that guesses the token after next. lumen verifies
the guess with the full model in a two-token pass and keeps the extra token when the guess was right
(58–76% of the time in prose, 86–90% in code).

The head is a separate 1.5 GB file that the shipped weights leave out. Fetch it once:

```sh
python3 scripts/qwen4_mtp_fetch.py ~/.slotstream/models/qwen38-flash-next-gs64
```

That downloads only the head's tensors from a public conversion (about 1.5 GB, a couple of minutes),
and writes `mtp.safetensors` and `mtp.json` into the model directory. The server finds them there.

- The head's routed experts are served from the expert pool like the trunk's, so it keeps only about
  50 MB resident.
- **What it buys here is little, so it is off unless you ask for it.** A verify pass needs about twice
  the experts of a plain step, and expert reads are what limits decode, so drafting saves only the GPU
  part of one step. Measured on a 36 GB M3 Max with the pack and direct reads (median of three 400-token
  runs, tokens/s, `off` → `auto`):

  | Pool | Prose | Code |
  |---|---|---|
  | 4.2 GB | 6.16 → 5.89 (−4.4%) | 6.48 → 6.74 (+4.0%) |
  | 6.7 GB | 7.38 → 7.15 (−3.1%) | 8.03 → 7.85 (−2.2%) |
  | 19.8 GB | 13.12 → 12.95 (−1.3%) | 13.44 → 13.30 (−1.0%) |

  The mean is −1.1% and the best case +4%. For code the governor stayed on and each round cost 1.8–2.0
  plain steps for 1.9 tokens, which is break-even; for prose the draft was accepted too rarely, so the
  governor mostly stayed off and its calibration and probe passes cost 1–2.5%.
- A governor measures milliseconds per token with and without drafts, on this machine with this text,
  and speculates only while that is clearly faster (it switches on below 0.95 of the plain time and
  off above 1.02); it looks at the other mode every few hundred tokens. Machines whose reads are
  slower or faster than this one's settle where it is better.
- **Not bit-identical to plain decoding.** The verify pass computes two rows at once, which adds
  floating-point sums in a different order than two single steps. Where two tokens are nearly tied
  the choice can differ, and the text diverges from there (a greedy run and a speculative run of the
  same prompt agree for the first dozens of tokens, then may differ in wording). The default `off` gives
  the exact plain-decode text. Because the governor switches by measured speed, the text of an `auto` run
  can also differ from run to run.
- Requests with a repetition, presence or frequency penalty, or under a grammar (tool calls,
  `response_format`), take plain steps.

## The expert pack

In the checkpoint an expert is nine ranges in nine tensors; a miss is nine scattered reads. The pack
rewrites the same bytes as one 2.7 MB record per expert, so a miss is one read. It costs another
68 GB of disk and is made once (49 s on an internal SSD):

```sh
lumen-server --pack-experts ~/.slotstream/models/qwen38-flash-next-gs64
```

or start the server with `LUMEN_MOE_PACK=build` and it builds the pack first if there is none.
Measured on a 36 GB M3 Max (median of three 300-token runs, direct reads, tokens/s, shards → pack, text
identical): 6.7 GB pool 6.69 → 7.15, 10 GB 8.06 → 8.96, 15 GB 10.39 → 11.15, 19.8 GB 11.96 → 12.95. A pack made for other
weights, or one cut short by an interrupted copy, is ignored and the shards are read as before.
Delete `experts.lpack` to get the disk back.
