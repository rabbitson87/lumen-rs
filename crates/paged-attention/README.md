# paged-attention — parked, pending an MLX rewrite

**This crate is excluded from the workspace and does not build.** It is kept on
disk because most of it is worth reusing, not because it works.

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
