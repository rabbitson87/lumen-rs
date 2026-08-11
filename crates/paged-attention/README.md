# paged-attention — parked, and measured

**This crate is excluded from the workspace and does not build.** It is kept on
disk because most of it is worth reusing, not because it works.

> **The revival was scoped, gated on a measurement, and the measurement said
> no.** See [Measured, 2026-08-11](#measured-2026-08-11) before re-deriving the
> case for reviving it — the short version is that on this workload paging can
> reclaim at most ~1% of process memory.

## Why it is parked

vLLM-style PagedAttention for Metal: block allocator, page table, scheduler,
and a 480-line MSL kernel. It was written against the Candle backend, which was
removed in task 006 — `candle-metal-kernels` supplied its `Buffer`, `Device`
and `ComputePipeline` types, and the only path that reached it from the server
(`PAGED_KV=1`) sat inside the GGUF branch, inside the Candle branch, which the
default build early-returned past. So it was already unreachable in a shipping
build before the removal; `docs/maintainer-workflow.md` recorded it as
"❌ WIP — Scaffolding only".

Deleting it outright would have thrown away real work. Leaving it in the
workspace would have left a crate that compiles against a dependency the
project no longer has.

## What is reusable as-is

| file | LOC | notes |
|---|---|---|
| `src/shaders/paged_attention.metal` | 480 | 5 kernels: `paged_attention_decode`, `_v2`, `write_kv_to_blocks`, `write_kv_to_blocks_f32_candle`, `copy_blocks`. Backend-agnostic MSL. |
| `src/scheduler.rs` | 270 | FCFS continuous batching + preemption. No Metal, no Candle. |
| `src/sequence.rs` | 177 | Per-request lifecycle. No Metal, no Candle. |
| `src/block_table.rs` | 124 | Logical→physical block mapping. Candle appears in one test line only. |

## What has to be redesigned, not ported

- **`src/block_allocator.rs` (451 LOC)** owns a pool of raw `MTLBuffer`s. Under
  MLX, blocks should be `mlx_rs::Array`s: MLX runs its own allocator, and the
  custom-kernel API takes Arrays rather than buffers. This is a redesign of the
  ownership model, not an import swap.
- **`src/metal/dispatch.rs` (371 LOC)** binds explicit `buffer(N)` slots against
  `candle_metal_kernels`. `lumen-mlx`'s kernel entry point is
  `mlx_sys::mlx_fast_metal_kernel`, which takes a kernel **body** and synthesizes
  the signature from the named input/output arrays. The `.metal` sources above
  are written as complete `kernel void …(device const T* q [[buffer(0)]], …)`
  declarations, so they need restructuring into that body form.

## Before reviving it, measure

PagedAttention solves fragmentation, copy-on-write forking, and high
concurrency. On Apple unified memory the first of those is a weaker motivation
than on discrete GPUs, and `lumen-mlx` already ships block-allocated KV
(`native_cache.rs`), single-copy shared prefixes (`prefix_cache.rs`) and a disk
tier (`kv_disk.rs`). `CLAUDE.md` itself rates PagedAttention low priority for
Mac serving.

So the revival should be gated on an A/B against the current `native_cache`
path — the same way the MLX embedding port was gated on parity with the Candle
model it replaced — rather than merged on the strength of the idea.

## Measured, 2026-08-11

That A/B exists: `cargo run --release -p lumen-mlx --features mlx-native
--example kv_concurrency_ab`. Run on M3 Max / 36 GiB against
`Qwen3.5-9B-MTPLX-Speed` (8 full-attention layers of 32, 4 KV heads, head_dim
256), at batch widths 1/2/4/8 across three prompt-length profiles.

Resident memory fits `per_seq * N + per_slot * allocated_slots` with no
intercept, R² ≥ 0.998 on all three profiles, and the profiles agree with each
other to ~10%: **~53 MB per sequence + ~67 KB per allocated cache slot**. The
per-slot figure matches the config arithmetic for an f32 cache
(8 x 4 x 256 x 2 x 4 B = 64.0 KB) and rules out bf16 (32.0 KB).

What paging would reclaim by replacing `KV_CACHE_STEP`-sized (256) blocks with
16-token ones, at N=8:

| prompt profile | resident | reclaimable | % of resident | % of process |
|---|---|---|---|---|
| short, 120-480 tok | 661.8 MB | 72.2 MB | 10.9% | **0.91%** |
| mixed, 200-3000 tok | 1,064.8 MB | 65.8 MB | 6.2% | **0.79%** |
| long, 2000-8000 tok | 2,960.3 MB | 35.4 MB | 1.2% | **0.35%** |

Short turns are the best case — rounding a 130-token sequence up to a 256-token
block wastes most of the block — and it is still 72 MB against a 7.9 GB process.

Two structural reasons it stays small, neither visible without measuring:

1. **40-63% of resident per-sequence memory is linear-attention conv/SSM
   state**, fixed-size per sequence regardless of context length. Paging is a
   full-attention KV technique; it cannot compact that at all. The naive
   "N x max_len" framing overstates the addressable share on a hybrid model.
2. **The binding constraint is prefill, not KV.** At N=8 long-context, prefill
   peaks 11.5 GB over baseline against a decode peak of 3.1 GB, because prefill
   materializes a `[1, prompt_len, vocab]` logits tensor — ~7.9 GB in a single
   allocation at 8K tokens and a 248,320 vocab. Paging does not touch it.

Throughput gives paging nothing to recover either: aggregate decode already
scales ~2.4x from N=1 to N=8, and residual active memory after `remove_seq` +
`clear_cache` was 0.0 MB at every width, so the existing path has no leak.

**If you are here to make serving fit in less memory, these are the bigger
levers** — both larger than paging, neither needing a custom kernel:

- **Store KV in bf16 rather than f32.** Halves the per-slot cost measured above
  (64 KB → 32 KB), ~1.5 GB at N=8 long-context versus paging's 35 MB. Not free:
  the f32 cast at `qwen3_5_moe.rs:2027-2029` mirrors the `qwen3_next` reference
  and attention accumulates in f32, so it needs its own parity gate.
- **Chunked prefill**, which caps the `[1, prompt_len, vocab]` peak that
  actually bounds usable context.

Re-run the harness before reviving this crate. The case would change if the
serving profile moved to very high concurrency with very short turns
(`--profile short --n 16,32`), or to a non-hybrid model where every layer holds
full-attention KV.
