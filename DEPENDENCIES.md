# External dependencies

This is an experimental Phase-1 release. A few dependencies need explicit setup
beyond `cargo build` because the workspace currently depends on local-path
versions of some libraries.

## Fork dependencies (auto-fetched)

`lumen-rs` consumes three upstream-fork dependencies pinned to specific
commit SHAs on `github.com/rabbitson87/<fork>` branches named
`lumen-rs-patches`. A clean `cargo build` fetches them automatically —
**no sibling clones required**.
The `mlx-native` feature triggers a longer first build (cmake +
fetched mlx C++ source) but works end-to-end without manual setup.

| Fork | Source | Why we fork |
|---|---|---|
| `rabbitson87/mlx-rs` | `oxiglade/mlx-rs` | `kestrel_flash_attn_bf16` primitive bindings + `memory` / `metal` modules for direct kernel dispatch. |
| `rabbitson87/mlx-c` | `ml-explore/mlx-c` | Kestrel C ABI (`lumen.h` / `lumen.cpp`) + array byte_offset / metal_buffer accessors. |
| `rabbitson87/mlx` | `ml-explore/mlx` | Kestrel custom primitive declarations + Metal-backend telemetry counters consumed by the perf-gate infrastructure. |

Each fork carries a single squashed `lumen-rs-patches` commit on top
of a known upstream baseline so future rebases are atomic. The
Cargo.toml pins to specific commit SHAs — upstream churn does not
affect lumen-rs until the SHA is explicitly bumped.

### Local override for fork development

The top-level `Cargo.toml` has a commented `[patch]` section that
redirects the git dep back to a sibling local clone
(`../../../mlx-rs/`). Uncomment it when iterating on the fork's
`lumen-rs-patches` branch:

```toml
[patch."https://github.com/rabbitson87/mlx-rs"]
mlx-rs  = { path = "../../../mlx-rs/mlx-rs" }
mlx-sys = { path = "../../../mlx-rs/mlx-sys" }
```

A `[patch]` for mlx-c / mlx is handled by mlx-c's CMakeLists.txt env
override: set `MLX_LOCAL_SOURCE_DIR=<your-mlx-checkout>` to redirect
the FetchContent call to a local mlx source tree.

## Model checkpoints (required at runtime)

Examples and the server expect environment variables pointing at local
MLX-quantized model checkpoints (`config.json` + `tokenizer.json` +
safetensors shards). None of the model weights are committed to this
repo. Useful env vars:

- `EMBEDDING_MODEL_ID` — Qwen3-Embedding-0.6B (MLX 8-bit quant)
- `MODEL_ID` / `LUMEN_GEMMA4_DIR` — Gemma 4 26B-A4B (MLX 3- or 4-bit)
- `LUMEN_QWEN35_SHARDS` — Qwen3.6-27B / Qwen3.5-30B-A3B (mxfp4)

See each example's module docs for the exact format expected.
