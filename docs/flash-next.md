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
  means fewer reads (hit rate 0.65 at a 6.7 GB pool, 0.88 at 19.8 GB on a 36 GB Mac).
- **Prefill** reads each layer's experts once per prompt (up to 8,192 tokens at a time), so it is
  limited by the GPU, not the SSD: about 230 tokens/s on an 18K-token prompt at any pool size.
- A turn with a few hundred new tokens still needs nearly every expert of every layer, so its first
  token costs one pass over the SSD (6–13 s) however short the turn is.

## Options

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_MOE_EXPERT_CACHE_GB` | `auto` | Size of the expert pool in GB. `auto` takes what the GPU's working set leaves after the 3.6 GB of resident weights and one prefill chunk. |
| `LUMEN_QWEN35_PREFILL_CHUNK` | 1024 | Tokens per prefill chunk. Prefill speed does not depend on it; its activations come out of the pool. |
| `LUMEN_MOE_IO_MODE` | `auto` | `direct` bypasses the page cache for expert reads, `buffered` uses it, `auto` uses it while the machine has more than 12 GB of memory to spare. |
| `LUMEN_MOE_IO_LANES` | 32 | Reader threads (1–64). |
| `LUMEN_MOE_PACK` | `auto` | The contiguous expert pack: `auto` uses `experts.lpack` when it is next to the weights, `off` never does, `build` makes it first. See below. |
| `LUMEN_QWEN4_MTP` | `auto` | The draft head for speculative decoding: `auto`, `on` or `off`. See below. |
| `LUMEN_QWEN4_MTP_FLOOR` | 0.4 | Share of drafts a conversation must keep getting accepted to keep speculating. |

## Speculative decoding (the draft head)

The model ships a small extra layer (the MTP head) that guesses the token after next. lumen verifies
the guess with the full model in a two-token pass and keeps the extra token when the guess was right
(83–89% of the time in prose).

The head is a separate 1.5 GB file that the shipped weights leave out. Fetch it once:

```sh
python3 scripts/qwen4_mtp_fetch.py ~/.slotstream/models/qwen38-flash-next-gs64
```

That downloads only the head's tensors from a public conversion (about 1.5 GB, a couple of minutes),
and writes `mtp.safetensors` and `mtp.json` into the model directory. The server finds them there.

- The head's routed experts are served from the expert pool like the trunk's, so it keeps only about
  50 MB resident.
- **What it buys here is small.** A verify pass needs about twice the experts of a plain step, and
  expert reads are what limits decode, so drafting saves only the GPU part of one step. Measured on a
  36 GB M3 Max at a fixed pool (median of three 400-token runs, tokens/s, off → on): 6.7 GB
  6.89 → 7.02, 10 GB 8.48 → 8.41, 15 GB 10.59 → 10.93, 19.8 GB 13.24 → 13.60.
- A governor measures milliseconds per token with and without drafts, on this machine with this text,
  and speculates only while that is clearly faster (it switches on below 0.95 of the plain time and
  off above 1.02); it looks at the other mode every few hundred tokens. Machines whose reads are
  slower or faster than this one's settle where it is better.
- **Not bit-identical to plain decoding.** The verify pass computes two rows at once, which adds
  floating-point sums in a different order than two single steps. Where two tokens are nearly tied
  the choice can differ, and the text diverges from there (a greedy run and a speculative run of the
  same prompt agree for the first dozens of tokens, then may differ in wording). Set
  `LUMEN_QWEN4_MTP=off` for the exact plain-decode text.
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
Measured on a 36 GB M3 Max (median of three 300-token runs, tokens/s, shards → pack, text
identical): 6.7 GB pool 5.75 → 6.83, 10 GB 7.53 → 8.22, 15 GB 9.73 → 10.86. A pack made for other
weights, or one cut short by an interrupted copy, is ignored and the shards are read as before.
Delete `experts.lpack` to get the disk back.
