# External dependencies

This is an experimental Phase-1 release. A few dependencies need explicit setup
beyond `cargo build` because the workspace currently depends on local-path
versions of some libraries.

## Fork dependencies (auto-fetched)

`lumen-rs` consumes four upstream-fork dependencies pinned to specific
commit SHAs on `github.com/rabbitson87/<fork>` branches named
`lumen-rs-patches`. A clean `cargo build` fetches them automatically —
**no sibling clones required for the default + qwen3_5_moe paths**.
The `mlx-native` feature triggers a longer first build (cmake +
fetched mlx C++ source) but works end-to-end without manual setup.

| Fork | Source | Why we fork |
|---|---|---|
| `rabbitson87/candle` | `huggingface/candle` | `pub fn clear_kv_cache` on Qwen3 + Metal / quantized helpers used by the lumen-rs Metal kernel integration. |
| `rabbitson87/mlx-rs` | `oxiglade/mlx-rs` | `kestrel_flash_attn_bf16` primitive bindings + `memory` / `metal` modules for direct kernel dispatch. |
| `rabbitson87/mlx-c` | `ml-explore/mlx-c` | Kestrel C ABI (`lumen.h` / `lumen.cpp`) + array byte_offset / metal_buffer accessors. |
| `rabbitson87/mlx` | `ml-explore/mlx` | Kestrel custom primitive declarations + Metal-backend telemetry counters consumed by the perf-gate infrastructure. |

Each fork carries a single squashed `lumen-rs-patches` commit on top
of a known upstream baseline so future rebases are atomic. The
Cargo.toml pins to specific commit SHAs — upstream churn does not
affect lumen-rs until the SHA is explicitly bumped.

### Local override for fork development

The top-level `Cargo.toml` has a commented `[patch]` section that
redirects the git deps back to sibling local clones (`../candle/`,
`../../../mlx-rs/`). Uncomment it when iterating on a fork's
`lumen-rs-patches` branch:

```toml
[patch."https://github.com/rabbitson87/candle"]
candle-core          = { path = "../candle/candle-core" }
candle-nn            = { path = "../candle/candle-nn" }
candle-transformers  = { path = "../candle/candle-transformers" }
candle-metal-kernels = { path = "../candle/candle-metal-kernels" }

[patch."https://github.com/rabbitson87/mlx-rs"]
mlx-rs  = { path = "../../../mlx-rs/mlx-rs" }
mlx-sys = { path = "../../../mlx-rs/mlx-sys" }
```

A `[patch]` for mlx-c / mlx is handled by mlx-c's CMakeLists.txt env
override: set `MLX_LOCAL_SOURCE_DIR=<your-mlx-checkout>` to redirect
the FetchContent call to a local mlx source tree.

## Test fixtures (3.4 GB, optional)

Several integration tests under `crates/lumen-model/tests/` consume
`.safetensors` weight dumps that are not committed to git (GitHub's
100 MB per-file limit). They are tied to the Qwen3.5 / Qwen3.6 MoE
backend (gated behind `--features qwen3_5_moe`); the default-feature
build does not need them.

### Download (clone-ers)

```bash
pip install huggingface_hub
python scripts/fetch_fixtures.py
```

This pulls the safetensors files from
[hsng95/lumen-rs-fixtures](https://huggingface.co/datasets/hsng95/lumen-rs-fixtures)
into `crates/lumen-model/tests/fixtures/` so the `--features qwen3_5_moe`
integration tests can run. Override with `LUMEN_FIXTURES_REPO=<USER>/<repo>`
if you maintain a private fork.

### Upload (maintainer)

```bash
hf auth login                       # one-time
hf upload hsng95/lumen-rs-fixtures \
  crates/lumen-model/tests/fixtures/layer0_moe_weights.safetensors \
  --repo-type dataset
hf upload hsng95/lumen-rs-fixtures \
  crates/lumen-model/tests/fixtures/layer0_linear_attn_weights.safetensors \
  --repo-type dataset
hf upload hsng95/lumen-rs-fixtures \
  crates/lumen-model/tests/fixtures/layer3_self_attn_weights.safetensors \
  --repo-type dataset
```

(Create the dataset repo first with `hf repos create hsng95/lumen-rs-fixtures --type dataset`
if it doesn't exist yet.)

## Model checkpoints (required at runtime)

Examples and the server expect environment variables pointing at local
MLX-quantized model checkpoints (`config.json` + `tokenizer.json` +
safetensors shards). None of the model weights are committed to this
repo. Useful env vars:

- `EMBEDDING_MODEL_ID` — Qwen3-Embedding-0.6B (MLX 8-bit quant)
- `MODEL_ID` / `LUMEN_GEMMA4_DIR` — Gemma 4 26B-A4B (MLX 3- or 4-bit)
- `LUMEN_QWEN35_SHARDS` — Qwen3.6-27B / Qwen3.5-30B-A3B (mxfp4, when
  building with `--features qwen3_5_moe`)

See each example's module docs for the exact format expected.
