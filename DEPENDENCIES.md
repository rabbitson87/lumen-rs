# External dependencies

This is an experimental Phase-1 release. A few dependencies need explicit setup
beyond `cargo build` because the workspace currently depends on local-path
versions of some libraries.

## Candle (required)

`Cargo.toml` points the four `candle-*` crates at a sibling directory:

```toml
candle-core           = { path = "../candle/candle-core" }
candle-nn             = { path = "../candle/candle-nn" }
candle-transformers   = { path = "../candle/candle-transformers" }
candle-metal-kernels  = { path = "../candle/candle-metal-kernels" }
```

We carry a small patch on `candle-transformers`:

- `candle-transformers/src/models/qwen3.rs` — make `Model::clear_kv_cache`
  `pub` so the stateless embedding loader (`lumen_model::qwen3_stateless`)
  can drop the KV cache between forward passes.

### Setup

Clone the candle fork next to this repo:

```bash
cd ..
git clone https://github.com/huggingface/candle.git
cd candle
# Apply the clear_kv_cache patch — see the diff in this repo's
# `docs/candle-patches/qwen3-clear-kv-cache.patch` (if not present,
# the one-line change is: drop the `fn` visibility on the existing
# `Model::clear_kv_cache` so it becomes `pub fn`).
```

A future revision of this repo will replace the path dependencies with
a pinned commit on a public fork, or with the upstream `candle` once the
patch lands there.

## Test fixtures (3.4 GB, optional)

Several integration tests under `crates/lumen-model/tests/` consume
`.safetensors` weight dumps that are not committed to git. They are
tied to the Qwen3.5 / Qwen3.6 MoE backend (gated behind
`--features qwen3_5_moe`); the default-feature build does not need
them. They will become downloadable from a HuggingFace Hub dataset in
a future revision of this repo.

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
