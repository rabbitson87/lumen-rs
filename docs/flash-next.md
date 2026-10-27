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
times 0.88 for the headroom other apps need; `LUMEN_MEMORY_LIMIT_GB` lowers it, and so does what
other apps hold at load, below) is split into the
3.6 GB of resident weights, a reserve for prefilling, and the expert pool, which gets the rest. The
reserve is 0.6 GB of buffers, about 1.1 MB per token of the prefill chunk, the KV cache of the
longest prompt the plan allows (30 KB per token in bf16, below; 60 KB in f32), and half the attention scores. So the longest prompt
and the pool trade against each other: `LUMEN_MAX_PROMPT_TOKENS` (the server's prompt cap, which the
plan also reads; `LUMEN_MAX_CTX` can only lower it) is 16,384 by default, and 65,536 takes about 1.5 GB out of the pool (3 GB in f32). The server rejects a
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
The table's plans were made with an f32 KV cache; bf16 (the default now) leaves the pool more.

### The KV cache in bf16 (`LUMEN_QWEN4_KV_BF16`, on by default)

The 12 attention layers (and the draft head's) keep their keys and values, and the sparse indexer its
keys, in bf16: 30 KB per token of context instead of 60, and attention runs in bf16 (the indexer pools
its keys in f32). Everything that holds a conversation is half the size: the plan's reserve for the
longest prompt, each session and resume point, an agent's system-prompt head (in memory and on disk).
This changes the numbers slightly, so `LUMEN_QWEN4_KV_BF16=0` brings back f32 (and a disk state saved
under one setting is never read under the other). Measured against f32 on this Mac:

| | f32 | bf16 |
|---|---|---|
| Same top token, 98K teacher-forced positions (Korean, English, code; 16K-token contexts) | (the reference) | 95.9% (95.3-96.2% at every depth) |
| KL from f32 / perplexity | - | 0.0093 nats / -0.05% |
| Expert pool at `LUMEN_MEMORY_LIMIT_GB=13` (24 GB class) | 6.16 GB | 6.90 GB |
| 16K-token prefill (two runs each, interleaved) | 60.8, 63.4 s | 56.7, 58.8 s |

For scale, the two 4-bit builds (group size 64 and 32) differ from each other by KL 0.20. Decode at
16K tokens reads 0.38 GB of KV per token instead of 0.77, about 1% of a token there.

### What other apps hold

The working set says how much one process may keep resident, not how much is free. With an editor,
a browser and chat apps open, a plan made from it alone takes memory those apps hold, and macOS makes
room by compressing them and writing them to swap. So at load the server also reads the kernel's page
counts (`host_statistics64`, the numbers `vm_stat` prints) and plans within

    free now = this process + free pages + file-backed pages + purgeable pages - headroom

when that is less than the working set's share. File-backed pages are copies of files, which the
kernel drops and reads again (speculative read-ahead is among them); purgeable pages are caches their
owners marked discardable. Taking either swaps nobody. Anonymous pages are not counted, inactive ones
included: they are another app's heap or GPU buffers, and taking them is compressing or swapping that
app. The headroom is 2 GB, or 5% of RAM on a Mac of more than 40 GB. The plan is MLX's, and the
process goes about 1.2 GB past it whatever the pool (measured at pools of 9.6, 13.0 and 19.2 GB:
0.55-0.75 GB beside MLX's buffers, and MLX's prefill peak ~0.5 GB over its soft limit); the rest is
for the free pages the kernel keeps and for other apps growing while the server runs.

`memory_pressure`'s "System-wide memory free percentage" is a different number: active + inactive +
free + speculative pages over RAM (`kern.memorystatus_level`), so every page that is neither wired nor
compressed, other apps' anonymous memory included. On the 36 GB M3 Max, read side by side:

| State | free | file-backed | purgeable | Reclaimable | Other apps hold | `memory_pressure` |
|---|---|---|---|---|---|---|
| Editor, browser, chat apps and an idle model server (10.7 GB compressed) | 0.7 | 6.9 | 0.3 | 7.9 GB | 30.8 GB | 55% = 21.3 GB |
| The same, the idle server stopped | 10.8 | 6.9 | 0.3 | 17.9 GB | 20.7 GB | 83% = 32.1 GB |

Only the pool shrinks: the chunk and the longest prompt stay what the working set gives (a busy 36 GB
Mac still takes 16K-token prompts, which the prompt guard reads from the plan). MLX's memory limit is
lowered to the same figure, or its buffer cache would grow back into the memory left to other apps. The
pool cannot change while the server runs, so a server started on a busy Mac keeps the smaller pool,
and decodes slower, until it is restarted. The log says so:

    [qwen4_exp] free now: 18.6 GB (free 14.1 + file-backed 4.5 + purgeable 0.0; other apps hold 20.0 GB that only compressing or swapping them frees) + this process 0.1 - headroom 1.9 = 16.7 GB for the plan (LUMEN_MEMORY_PLAN_RECLAIMABLE=0 plans without it)
    [qwen4_exp] GPU may use 26.5 GB, 16.7 GB of it free now: expert pool 9.64 GB, prefill chunk 1024 tokens, prompts up to 16384 tokens (...). Pool 9.6 GB (other apps hold memory: 9.8 GB below the working set); close apps and restart for a bigger pool

A process that has just exited gives its memory back about a second later (MLX's buffers are wired:
2.8 GB was reclaimable when a loaded server's process had exited, 26 GB a second later), so a server
restarted at once would plan for almost nothing. While a reading would cut the plan, the server takes
it again every half second until it stops growing, for at most 3 s; memory other apps hold does not
grow back, so on a busy Mac that costs the load one extra reading.

An explicit `LUMEN_MOE_EXPERT_CACHE_GB` is lowered to what is free now in the same way, with a line
that says so. When not even the smallest pool (2.83 GB) fits beside what other apps hold, the server
uses it anyway, as a tight Mac always has, and the line says to expect swapping.
`LUMEN_MEMORY_PLAN_RECLAIMABLE=0` plans from the working set alone, as before.

Measured on the 36 GB M3 Max with those apps open: a fresh server per run, defaults (no pool size, no
memory limit), a 14,962-token prompt of real text, then 300 tokens; runs stopped when swap grew by more
than 1.5 GB. The headroom was 1.93 GB then (5% of RAM; it is now at least 2 GB).

| Plan | Free now at load | Pool | Swap growth | Pages compressed | Peak footprint | First token | Decode |
|---|---|---|---|---|---|---|---|
| Free now (run 1) | 18.6 GB | 9.64 GB | 0 | 5.0 GB | 17.9 GB | 94.6 s | 2.9 tok/s |
| Free now (run 2) | 21.9 GB | 12.97 GB | 0 | 7.2 GB | 21.3 GB | 97.2 s | 3.1 tok/s |
| Working set (run 1) | 24.1 GB | 19.42 GB | +1.66 GB in 29 s, stopped | 21.8 GB | 25.3 GB | none | - |
| Working set (run 2) | 26.9 GB | 19.42 GB | +1.82 GB in 76 s, stopped | 7.2 GB | 26.9 GB | none | - |

The working-set plan grew swap by 1.7-1.8 GB before its first token, while it was still loading the
pool and reading the prompt; the plan from what was free swapped nothing. It did not stop macOS from
compressing other apps (their anonymous memory was 11.9 GB when the first run started and 7.9 GB when
the second did): the kernel compresses inactive anonymous pages before it has dropped every file page. Decode speeds here are
this machine's at the time (other GPU work running; 3.6 tok/s at a 19.2 GB pool with nothing in its
way, against 13 on an idle machine), so they compare the runs with each other, not with the tables
above. A real 16 or 24 GB Mac was not measured.

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

### An agent's system prompt on disk

Requests with tools (agents) keep the head of their system prompt (the system message and the rendered
tools) in memory, so the next conversation with the same system prompt feeds only its own turn. The
disk KV tier also writes that head to disk (`LUMEN_KV_DISK_DIR`, default `~/.cache/lumen/kv`), once per
system prompt, after the reply. A restart, or the memory plan above dropping the head, then reads it
back instead of prefilling it again.

The disk tier is on by default, within three limits:

| Variable | Default | `0` means |
|---|---|---|
| `LUMEN_KV_DISK_MAX_GB` | 10: past it the least recently used entries go | unlimited |
| `LUMEN_KV_DISK_TTL_SECS` | 2592000 (30 days since an entry was last used) | entries never expire |
| `LUMEN_KV_DISK_MIN_FREE_GB` | 10: no writes while the store's volume has less free | no floor |

Under the free-space floor nothing is deleted: what is on disk stays and is still read, and writes
resume once there is room (one log line when they stop, one when they resume). `LUMEN_KV_DISK=0` (or
`false`, `off`, `no`) turns the tier off; the desktop app's DISK KV CACHE card sends the same switch.

Measured with a 12.3K-token agent prompt (instructions, two project documents, 26 tool schemas) at a
6.7 GB pool, medians of N=3, every reply identical to the cold one:

| Next conversation with the same system prompt | Before | Now |
|---|---|---|
| The first one (cold) | 43.7 s | 40.6 s |
| Head in memory | 2.6 s | 3.0 s (2.9-3.0) |
| After the head was dropped from memory | 39.1 s (prefilled again) | 2.8 s (read back in 0.25-0.4 s) |
| After a server restart | 44.2 s (prefilled again) | 3.5 s |
| 24 GB class (`LUMEN_MEMORY_LIMIT_GB=13`): a second agent's system prompt pushed the head out | 40.4 s | 2.8 s |
| 24 GB class, after a restart | 43.3 s | 3.1 s |

The head of 12,244 tokens is 795 MB on disk. The request that writes it spends 0.1 s reading it back
and 0.1-0.4 s serializing it after its last token (its final chunk comes that much later); the file
write runs on the disk tier's own thread and lands 0.3-1.4 s later. A request that starts while that
write is still going shares the SSD with it (the in-memory hit right after the first request: 2.9-3.0 s
against 2.6 s). Later requests only mark the file used. On a 24 GB Mac, a head read back for a prompt
the memory plan sent without one is used for that prompt and released again (it stays on disk), so
decode keeps its speed.

The head sits where the cold request with that system prompt already cuts its prefill (the head, then
the rest), so forking it computes exactly what the cold request computes. On this mixture-of-experts
model that matters: the same prompt cut at a different place (the 1,024-token chunk grid instead of the
head) gave a different greedy reply, and moving the head down to the grid would make every hit feed up to
a chunk again (10.3 s instead of 2.6 s for a hit, measured). That is also why only heads are written: a
conversation's latest turn is not a cut its next turn's cold prefill makes. The cold request now feeds
the head and the rest in one pass, so each layer's experts are read once instead of twice (40.6 s against
42.5 s in two calls). A conversation that continues after a restart resumes from the head and feeds its
later turns again; a prompt the memory plan sends without a head writes none.

### An agent's later rounds (`LUMEN_PREFIX_EXACT_REUSE`, on by default)

A tool round resumes only from the head or a point a whole number of chunks past it (kept instead of the
previous round's full-prompt snapshot) and re-feeds the rest on that grid, so every round is exactly its
cold prefill. `LUMEN_PREFIX_EXACT_REUSE=0` goes back to extending the previous round's full-prompt
snapshot, which is cut where the previous prompt ended, a place the round's own cold prefill (the head,
then a chunk at a time from there) does not cut, so its reply can differ. Measured on three 6-round agent tasks (12.3K-token head, real tool
results, the conversation fixed so every variant got the same inputs; 6.7 GB pool, greedy):

| | Replies that differ from the round's cold prefill | Prefill of rounds 2-6, mean / p90 | 3 sessions |
|---|---|---|---|
| Snapshot (`=0`) | 4 of 18 (each a different tool call or arguments) | 10.8 s / 15.9 s | 505 s |
| Exact (default) | 0 of 18 | 10.8 s / 11.6 s | 486 s |
| Every round cold | (the reference) | 51.5 s / 66.9 s | 1,138 s |

Exact re-feeds 58-979 more tokens than the snapshot path (mean 389). That costs nothing when the round feeds a
few hundred tokens anyway (both read every layer's experts once), but a round with under ~100 new tokens
goes from 2.7-4 s to 10.4 s, since re-feeding from the head reads them all. The snapshot path loses its
snapshot when a reply's text before its tool call is only a newline (that turn re-renders as different
tokens) and re-feeds from the head, up to 6,099 tokens and 24 s, where exact keeps its point. Note that the
cold prefill with the cache on (cut at the head) is not the cache-off prefill (one cut per chunk from the
start): those two differed in 10 of the 18 rounds, so exact matches the former, not the latter.

### Several agents at once (`LUMEN_JOINT_PREFILL`, on by default)

The server serves one request at a time. When others are waiting, each waiting agent round that extends a
conversation the prefix cache holds (a tools request) is fed together with the one being served: one pass
over the layers feeds them all, so each layer's experts are read from the SSD once instead of once per
request. Each request still gets its own feed, from its own snapshot and cut into its own chunks, so its
reply is the one it gets alone. It takes the result only if, when its turn comes, its sequence is still the
untouched fork of that snapshot; otherwise it computes its own, and the result fed ahead is dropped. Up to
8,192 tokens are fed together, and only as much as the memory plan holds without dropping anything.
`LUMEN_JOINT_PREFILL=0` feeds each request on its own. Both OpenAI-format (`/v1/chat/completions`) and
Anthropic-format (`/v1/messages`) requests are fed ahead, streamed or not; requests with images are not
(they do not use the prefix cache).

Measured with the three 6-round agent tasks above sent at once (one second of tool time between a reply and
the next request; 2.9 GB pool, greedy, streamed):

| | 3 sessions | Mean wait for the first token | Replies that differ from serving one request at a time |
|---|---|---|---|
| One request at a time | 623.6 s | 25.5 s | (the reference) |
| Three at once, `=0` | 609.0 s | 88.1 s | 0 of 18 |
| Three at once (default) | 530.1 s (-13%) | 76.2 s | 0 of 18 |

The wait includes the time a request spends queued behind the others. Without joint feeding, three at once
is barely faster than one at a time: the server does the same work, only without the second between rounds.
Joint feeding is what removes work: eight times two waiting rounds were fed together, and seven of those
pairs took 20-29 s against 30-38 s apart (about 10 s saved each); two short rounds (74 and 58 tokens) took
4.9 s against 7.2 s apart. Prefill took 77 s less in all; decode is the same (4.8 tok/s).

Short feeds fed together each take the expert pool, as they would alone: staging a whole layer for them
(as a long prefill does) made that same pair take 11.4 s. And a request whose result was fed ahead plans
no more memory for it, since the result is in memory already: counted again, it released the agent's
system-prompt head on a busier machine, and that conversation's next round prefilled cold (60 s).

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
| `LUMEN_MOE_EXPERT_CACHE_GB` | `auto` | Size of the expert pool: `auto`, or a size such as `4g`, `8GB`, `12.5g`, `512m` (a plain number is GB). `auto` takes what the GPU's working set (or what other apps leave free at load, when that is less) leaves after the 3.6 GB of resident weights and the prefill reserve. A size that does not fit beside them (at the longest prompt the plan allows) is lowered to what does, and the log says so; a shorter `LUMEN_MAX_PROMPT_TOKENS` makes room. Measured on a 36 GB M3 Max with the pack and direct reads, 300 tokens: 4.2 GB 6.1 tok/s, 6.7 GB 7.2, 10 GB 9.0, 15 GB 11.2, 19.8 GB 13.0. |
| `LUMEN_QWEN35_PREFILL_CHUNK` | 1024 (512 under 11 GB) | Tokens per prefill chunk. Prefill speed does not depend on it; its activations come out of the pool. |
| `LUMEN_MAX_PROMPT_TOKENS` | 16384 (8192 under 11 GB) | Longest prompt; the KV cache for it is planned out of the pool. |
| `LUMEN_MEMORY_LIMIT_GB` | 32 | Cap on MLX's memory in GiB; the pool is planned within it. |
| `LUMEN_MEMORY_PLAN_RECLAIMABLE` | on | Plan the pool within what is free at load: this process, free, file-backed and purgeable pages, less 2 GB or 5% of RAM (see What other apps hold). MLX's memory limit is lowered to the same figure. `0` plans from the working set alone, and the system makes room by compressing and swapping other apps. |
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
