# lumen-rs

> Experimental in-process LLM inference server in Rust for Apple Silicon.
> OpenAI-compatible HTTP endpoints, custom Metal kernels, MLX-quantized
> weights, single binary. No Python subprocess, no PyO3 hops.

**Status: alpha / experimental.** This is a working-in-progress snapshot of
a personal research project. Apple Silicon only. Currently validated:

- `/v1/embeddings` via Qwen3-Embedding-0.6B (MLX 8-bit)
- `/v1/chat/completions` via Gemma 4 26B-A4B MoE (MLX 3- or 4-bit)

Ships in two forms:

1. **Desktop app (`lumen-app`)** — a Tauri 2.x app that wraps the server as
   a sidecar process, exposes every env-var knob as a typed UI field, and
   self-updates via GitHub Releases. Recommended for end users who just
   want to run a local LLM.
2. **CLI server (`lumen-server`)** — the same engine as a plain HTTP
   binary you launch with env vars. Recommended for headless deployments
   and library/research use.

Other model paths exist in the codebase and should be treated as exploratory
(may not work without local tweaks):

- **Qwen3.5 30B-A3B / Qwen3.6 27B MoE** — full Candle backend including a
  256-expert MoE assembly + native MLX runner. Opt-in via
  `--features qwen3_5_moe` (off by default). Mostly working; see source
  comments and the `LUMEN_QWEN35_SHARDS` env var.
- **GGUF Gemma** via candle's gguf loader (CPU/Metal).
- **Candle Qwen** legacy path.

---

## Table of contents

- [Requirements](#requirements)
- [Desktop app](#desktop-app) — install + run the GUI
- [Install](#install) — CLI / library / source build
- [Configure models](#configure-models)
- [Run the server](#run-the-server)
- [API reference](#api-reference)
- [Run the bundled examples](#run-the-bundled-examples)
- [Environment variables](#environment-variables)
- [Architecture](#architecture)
- [Build flags / Cargo features](#build-flags--cargo-features)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Repo layout](#repo-layout)
- [License](#license)

For a step-by-step walkthrough from a clean clone (build → models →
server → benches → fixture download → A/B switches), see
[**docs/getting-started.md**](docs/getting-started.md). The rest of
this README is reference material.

For maintainers (and AI agents operating on this repo): see
[**docs/maintainer-workflow.md**](docs/maintainer-workflow.md) for the
commit-message conventions, fork-SHA bump procedure, validated-path
table, perf-regression gates, and feature-flag policy that the `main`
branch is currently maintained against. Read it before authoring a
commit on this repo.

---

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon (M1 / M2 / M3 / M4) |
| OS | macOS 14+ |
| Toolchain | Rust 1.85+ (edition 2024). Install via `rustup`. |
| Disk | ~50 GB free for a full Gemma 4 26B MoE checkpoint + workspace target/ |
| Memory | Embedding alone: ~1 GB. Gemma 4 26B-A4B MLX 4-bit: ~22 GB unified memory at steady state. |

Optional (only if running mlx-lm parity benches or regenerating fixtures):

- Python 3.10+ with `mlx-lm`
- A separate clone of the [mlx](https://github.com/ml-explore/mlx) source
  pointed to by `MLX_LOCAL_SOURCE_DIR` (used by some benchmark scripts)

---

## Desktop app

The Tauri-based GUI (`crates/lumen-app`) is the recommended onboarding
path. It wraps `lumen-server` as a sidecar, manages model downloads from
HuggingFace Hub, exposes every memory/context/generation knob the engine
reads, and self-updates from GitHub Releases.

### Install from a release

1. Open the [latest GitHub Release](https://github.com/hsng95/lumen-rs/releases/latest).
2. Download the `aarch64` `.dmg`. **Apple Silicon only** — MLX is
   ARM64-native and refuses to build on x86_64, so there is no Intel
   Mac bundle.
3. Drag `Lumen.app` to `/Applications` and launch it. The first run
   prompts macOS to verify the developer signature — accept it once.
4. On the **Models & Server** tab, pick a recommended model from the
   dropdown and hit **Download**. Wait for completion, then **Use** →
   **Start**.

The app handles the rest:

- Metal memory caps are auto-tuned to the active model + context size
  (wired = byte-precise model size via `LUMEN_WIRED_LIMIT_BYTES`, cache
  = flat 2 GB, memory = ceil(model + 2 + ctx/8K)).
- The **API** tab generates copy-pasteable `curl` examples in
  OpenAI-style or Claude-style with your configured API key already
  interpolated.
- **Doctor** runs preflight diagnostics (RAM / disk / port / HF
  reachability) and auto-opens when anything is `blocked` or
  `degraded`.
- **Update** checks GitHub Releases on demand; signed installs
  atomically swap both the `.app` bundle and the sidecar
  `lumen-server` binary.

### Build the desktop app from source

```bash
cd crates/lumen-app/frontend
npm install
cd ..
cargo install tauri-cli --version "^2"   # if not already installed
cargo tauri dev                          # hot-reloading dev mode
# or
cargo tauri build                        # release .app + .dmg bundle
```

The desktop app is excluded from `default-members` in the workspace
`Cargo.toml`, so `cargo build` at the repo root still builds only the
core crates (no Tauri / webview dependencies). Build it explicitly with
`cargo build -p lumen-app` when you do need the Rust side without the
frontend.

For release maintainers (signing keys, GitHub Actions workflow,
schema-migration policy), see
[crates/lumen-app/docs/release.md](crates/lumen-app/docs/release.md).

---

## Install

### 1. Clone the repo plus the patched candle fork

The workspace currently depends on a sibling `candle/` directory with a
one-line patch on `candle-transformers` (see [DEPENDENCIES.md](DEPENDENCIES.md)
for the exact change).

```bash
cd ~/your-projects/
git clone <THIS_REPO_URL> lumen-rs

# Sibling candle checkout — required because Cargo.toml uses path = "../candle".
git clone https://github.com/huggingface/candle.git
# Apply the clear_kv_cache patch — see DEPENDENCIES.md for the diff.
```

Your tree should look like:

```
your-projects/
  ├── candle/         (the patched fork)
  └── lumen-rs/       (this repo)
```

### 2. Build

```bash
cd lumen-rs
cargo build --release                          # default features: Metal + TurboQuant GPU
cargo build --release --features mlx-native    # add the native Gemma 4 MLX backend
```

The first build downloads dependencies and compiles ~250 crates. Allow
~5–10 minutes on a clean M-series machine.

### 3. Smoke test the embedding stack (no checkpoint setup required)

```bash
cargo test -p lumen-metal --release --test affine8_parity
# expected: 7 passed; 0 failed
```

---

## Configure models

`lumen-rs` does not bundle any model weights. You point env vars at local
directories holding MLX checkpoints (`config.json` + `tokenizer.json` +
`model*.safetensors`).

### Embedding model

Download a Qwen3-Embedding MLX 8-bit checkpoint. For example:

```bash
huggingface-cli download \
  mlx-community/Qwen3-Embedding-0.6B-8bit-mlx \
  --local-dir ~/models/qwen3-embedding-0.6b-8bit
```

(Substitute any MLX 8-bit Qwen3-Embedding fork that suits you.)

Then:

```bash
export EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit
```

### Chat model (Gemma 4 26B-A4B MoE)

Download an MLX 4-bit or 3-bit Gemma 4 26B-A4B checkpoint:

```bash
huggingface-cli download \
  mlx-community/gemma-4-26b-a4b-mlx-4bit \
  --local-dir ~/models/gemma-4-26b-a4b-mlx-4bit
```

Then point either of:

```bash
export MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit
# or
export LUMEN_GEMMA4_DIR=~/models/gemma-4-26b-a4b-mlx-4bit
```

---

## Run the server

```bash
# Both endpoints enabled:
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit \
  cargo run --release --features mlx-native --bin lumen-server
```

The server listens on `127.0.0.1:8080` by default. Override with `PORT` /
`HOST` env vars if needed.

### Embedding-only mode

If you only want the embedding endpoint (no Gemma 4 checkpoint required):

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release --bin lumen-server
```

The `/v1/chat/completions` route will return 503 when no chat model is
configured; `/v1/embeddings` will work.

---

## API reference

### `POST /v1/embeddings`

OpenAI-compatible. Single string or array of strings.

```bash
curl -s localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "input": ["hello world", "안녕 세계"]
  }' | jq '.data[0].embedding | length'
# 1024
```

Response:

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.012, -0.034, ...] },
    { "object": "embedding", "index": 1, "embedding": [-0.001,  0.027, ...] }
  ],
  "model": "qwen3-embedding-0.6b",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

All vectors are L2-normalized and 1024-dimensional.

### `POST /v1/chat/completions`

OpenAI-compatible. Non-streaming greedy decode (sampling lands in a
follow-up).

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b",
    "messages": [
      {"role": "user", "content": "What is the capital of South Korea?"}
    ],
    "max_tokens": 64
  }' | jq .
```

### `POST /v1/completions`

OpenAI-compatible legacy completions endpoint. Same backend as
`/v1/chat/completions` but with a plain `prompt` field.

### `GET /v1/models`

Returns the loaded model identifiers.

---

## Run the bundled examples

### Embedding smoke test

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release -p lumen-model --example embedding_smoke
```

Loads the model, embeds three Korean+English phrases, prints timings, RSS,
norm validation, and cosine similarities.

```
[smoke] model loaded — dim=1024 max_seq_len=32768
[smoke] RSS: 6 MB → 890 MB (+884 MB at load)
[smoke] embed b=3 over 5 iters: median=19.35ms min=18.72ms max=22.35ms
[smoke] OK: semantic ordering preserved
```

### Embedding quality eval

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release -p lumen-model --example embedding_quality
```

Embeds a 25-item KR/EN multi-domain corpus (KBO baseball, NBA basketball,
programming languages, Korean cities, Korean food). Reports retrieval
metrics:

```
[quality] embed b=25 in 251ms (10.05ms/item)
[quality] P@1 = 0.960   P@3 = 0.880   MRR = 0.980
[quality] per-category MRR:
  baseball_kbo       (5 items)  MRR = 0.900
  basketball_nba     (5 items)  MRR = 1.000
  city_korea         (5 items)  MRR = 1.000
  food_korean        (5 items)  MRR = 1.000
  programming        (5 items)  MRR = 1.000
```

### Compare the qmv_fast vs naive 8-bit kernel

```bash
# qmv_fast (default — cooperative simdgroup):
cargo run --release -p lumen-model --example embedding_smoke

# Naive 1-thread/output (slow path; useful for A/B testing):
LUMEN_AFFINE8_NAIVE=1 \
  cargo run --release -p lumen-model --example embedding_smoke
```

On an M3 Max you should see ~19 ms/batch with qmv_fast vs ~35 ms with
naive, with identical cosine ordering — proof that the kernel-level
optimization preserves model quality.

---

## Environment variables

### Required (for the relevant endpoint)

| Var | Purpose |
|---|---|
| `EMBEDDING_MODEL_ID` | Local path or HF Hub repo id to a MLX 8-bit Qwen3-Embedding checkpoint. |
| `MODEL_ID` / `LUMEN_GEMMA4_DIR` | Local path to a MLX Gemma 4 26B-A4B checkpoint (either env works). |

### Optional knobs

| Var | Purpose |
|---|---|
| `PORT`, `HOST` | HTTP listen address (defaults `127.0.0.1:8080`). |
| `LUMEN_MODE` | `mlx` \| `candle` \| `auto`. Selects the backend mode. Defaults to `mlx` when built with `mlx-native` (default); `candle` otherwise. |
| `LUMEN_MLX_BACKEND` | `native` \| `pyo3` \| `subprocess`. Picks the mlx runner. Defaults to `native` under the `mlx-native` feature (+57% tok/s vs Candle, 33× at PROMPT_LEN=2048 on Qwen3.6-35B-A3B-mxfp4). |
| `LUMEN_AFFINE8_NAIVE=1` | Force the naive 8-bit GEMM kernel path (A/B testing). |
| `LUMEN_GEMMA4_PREFILL_SYNC=0` | Disable the explicit eval-sync after prefill (advanced; see source comments). |
| `BATCHED_ENGINE=1` | Continuous-batching scheduler for the GGUF and Qwen3.6 backends. |
| `KESTREL_GEMMA4_CUSTOM_FLASH_ATTN=0` | Opt-out of the custom flash-attention primitive (default on). |
| `KESTREL_GEMMA4_PER_STEP_LATENCY=1` | Dump per-step latency table at the end of generation. |

A full list of advanced flags lives in source-level docstrings under
[crates/lumen-mlx/src/env_state.rs](crates/lumen-mlx/src/env_state.rs).

---

## Architecture

Three layered components separated by traits so the codec stays portable:

1. **TurboQuant codec** (`lumen-core`) — pure Rust, hardware-agnostic.
   Lloyd-Max scalar quantization + random orthogonal rotation + 1-bit
   QJL residual. Implements the [TurboQuant ICLR'26
   paper](https://arxiv.org/abs/2504.19874).
2. **KV-cache strategies + model code** (`turboquant-cache`,
   `lumen-model`, `lumen-mlx`). Candle-based models for embedding and
   GGUF/Gemma paths; native MLX-rs assembly for Gemma 4 26B-A4B and the
   shared Qwen3 native runner.
3. **Serving** (`lumen-server`). atomic_http-based OpenAI-compatible
   HTTP server. Optional continuous-batching scheduler.

Two pieces of custom kernel work that the public release covers:

- **`affine8_qmv_fast_bf16`** in `crates/lumen-metal/src/shaders/affine8.metal`
  — cooperative simdgroup 8-bit MLX-format GEMM. NSG=2 × RPS=4
  (8 outputs per threadgroup, 32-lane K-dimension split). Bit-parity
  with a CPU reference; ~45 % latency reduction over the naive
  1-thread/output kernel at Qwen3-Embedding shapes.
- **`kestrel_flash_attn_bf16`** mlx Primitive. Bit-near-identical
  (max|Δ|=1.95e-3) to `mlx::fast::sdpa`. Registered as a first-class
  mlx Primitive — keeps the kernel in mlx's own command-buffer
  batching, avoiding the bridge-dispatch cost (~30 ms/step when the
  pattern is violated).

---

## Build flags / Cargo features

| Crate | Feature | Effect |
|---|---|---|
| `lumen-model` | `metal` *(default)* | Metal backend on `candle-*`. |
| `lumen-model` | `turboquant` *(default)* | Candle TurboQuant attention. |
| `lumen-model` | `turboquant-gpu` *(default)* | GPU dispatch for affine quantization kernels. |
| `lumen-model` | `paged-kv` *(default)* | PagedAttention KV-cache scaffolding. |
| `lumen-model` | `qwen3_5_moe` | Qwen3.5 30B-A3B / Qwen3.6 27B MoE backend (Candle path). Off by default; opt-in for chat completions on these checkpoints. |
| `lumen-mlx` | `mlx-native` | Pure-Rust Gemma 4 26B-A4B path via mlx-rs. **Required for `/v1/chat/completions`.** |
| `lumen-mlx` | `mlx-pyo3` | PyO3 / mlx-lm subprocess fallback (development only). |
| `lumen-server` | `qwen3_5_moe` | Forwards the lumen-model qwen3_5_moe feature. |

Typical command lines:

```bash
# Minimum: embedding-only server.
cargo build --release

# Embedding + Gemma 4 chat (recommended dev / prod build).
cargo build --release --features mlx-native

# Add the Qwen3.5 / Qwen3.6 MoE backend.
cargo build --release --features mlx-native --features lumen-server/qwen3_5_moe
```

---

## Performance

Indicative numbers on an Apple M3 Max:

| Workload | Latency | Notes |
|---|---|---|
| Embedding b=3, len≈18 tokens | 19.4 ms/batch | naive 8-bit kernel: 35.5 ms (−45 %) |
| Embedding b=25 quality eval | 251 ms (≈10 ms/item) | P@1 = 0.960 on the labelled corpus |
| Gemma 4 26B-A4B decode | 18.8 ms/step | mlx default sdpa: 19.9 ms (custom flash-attn −5 %) |
| Gemma 4 26B-A4B prefill 4 k tokens | ~4.0 s | Full path including JIT-compile warmup |
| Qwen3.6-35B-A3B-mxfp4 decode (N=1, **mlx-native default**) | 13.94 ms/step p50 | **71.6 tok/s** — gather_qmm reads only top-K experts |
| Qwen3.6-35B-A3B-mxfp4 decode (N=1, Candle) | 22.0 ms/step p50 | 45.5 tok/s (−36 % step latency / +57 % tps when switched to mlx) |
| Qwen3.6-35B-A3B-mxfp4 decode (PROMPT_LEN=2048, mlx-native) | 14.85 ms/step p50 | 67.3 tok/s |
| Qwen3.6-35B-A3B-mxfp4 decode (PROMPT_LEN=2048, Candle) | 486 ms/step p50 | 2.0 tok/s — Candle SDPA does not scale to long KV |
| Qwen3.6-35B-A3B-mxfp4 decode (N=2 CB, Candle) | 81.9 ms/step | aggregate 24.4 tok/s (+17 % over N=1 Candle) |

### Why mlx-native is the default

At single-batch decode the bottleneck is bandwidth, not FLOPs. A 35B-A3B
MoE *should* read only the top-K active experts per step (~0.5 GB) instead
of all 256 (~17 GB). The mlx runner achieves this via `gather_qmm`, which
fuses expert routing + quantized matmul into a single scatter-gather
kernel. Candle's per-expert loop pays close to full-model bandwidth, which
shows up as a 1.6× slowdown at short prompts and a 33× cliff at
PROMPT_LEN=2048 once attention KV joins the read budget.

The server defaults to `LUMEN_MODE=mlx` + `LUMEN_MLX_BACKEND=native`
whenever the `mlx-native` feature is compiled in. Pass
`LUMEN_MODE=candle` to opt into the multi-tenant CB path (Candle still
wins on aggregate throughput at N≥4 because mlx-rs single-tenant
microbatching does not fan out).

### Kernel fusion impact (Qwen3.6-35B-A3B-mxfp4, N=1, Candle backend only)

| Config | p50 step latency | aggregate tps | vs fused |
|---|---|---|---|
| Fused (Candle default) | 48.4 ms / 57.4 ms | 20.8 / 13.9 tok/s | baseline |
| All `LUMEN_DISABLE_*=1` | 66.3 ms / 77.7 ms | 15.1 / 10.3 tok/s | **+27 % slower** |

(Two runs shown to highlight thermal sensitivity; p50 is the stable metric.)
Fused kernels covered: flash-attn, residual+RMSNorm, input RMSNorm, dense MLP
residual, MoE gate/up/SiLU/mul, MoE weighted-sum. These apply to the Candle
path only — the mlx-native runner relies on mlx's `gather_qmm` + custom
`kestrel_flash_attn_bf16` primitive instead.

### Resident memory

- Qwen3-Embedding 8-bit: ~900 MB GPU footprint (vs ~1.4 GB for plain bf16,
  −37 %).
- Gemma 4 26B-A4B MLX 4-bit: ~22 GB unified memory at steady state.
- Qwen3.6-35B-A3B-mxfp4: ~22 GB unified memory at steady state.

---

## Troubleshooting

**`error: failed to load source for dependency`candle-core`** — you don't
have the candle fork at `../candle/`. See [DEPENDENCIES.md](DEPENDENCIES.md).

**Server starts but `/v1/embeddings` returns 503** — `EMBEDDING_MODEL_ID`
is not set or the path doesn't exist. Confirm `ls $EMBEDDING_MODEL_ID`
shows `config.json` + `tokenizer.json` + safetensors shards.

**`/v1/chat/completions` returns 503** — `MODEL_ID` / `LUMEN_GEMMA4_DIR`
is unset, or you built without `--features mlx-native`.

**Slow embedding latency (~35 ms instead of ~19 ms)** — the `qmv_fast`
kernel needs `in_features % 512 == 0 AND out_features % 8 == 0`. The
Qwen3-Embedding-0.6B shapes (1024 / 3072 in; 512 / 1024 / 3072 / vocab out)
satisfy both, so this should not trigger. If you see naive-kernel speed,
unset `LUMEN_AFFINE8_NAIVE`.

**`thread panicked at 'metal command buffer not enqueued'`** — known
intermittent issue when interleaving Candle and mlx kernels on the same
queue. Restart the server; this happens during shutdown for the most
part.

**Out-of-memory on Gemma 4** — the 26B-A4B model needs ~22 GB unified
memory. On 24 GB machines, switch to the 3-bit MLX variant
(`mlx-community/gemma-4-26b-a4b-mlx-3bit`) which fits in ~16 GB.

---

## Repo layout

```
crates/
  lumen-core/         pure-Rust TurboQuant codec (Lloyd-Max + QJL, hardware-agnostic)
  lumen-metal/        Metal compute kernels: affine 3/4/8-bit quant GEMM,
                      MXFP4, flash-attn, rms_norm, silu_mul, sampling
  lumen-mlx/          MLX-native Gemma 4 26B-A4B MoE backend + custom mlx
                      primitives (kestrel_flash_attn_bf16) + bridge crates
  lumen-model/        candle-based model assemblies (Gemma, Gemma-GGUF,
                      Qwen, Qwen3-Embedding) + KV-cache strategies
  lumen-server/       atomic_http-based OpenAI-compatible HTTP server
                      (/v1/embeddings, /v1/chat/completions, /v1/completions)
  turboquant-cache/   KVCache trait + SimpleCache / PagedCache scaffolding
  paged-attention/    PagedAttention scaffolding (WIP)

deploy/               example .env + launchd plist for macOS service install
examples/             end-to-end demo binaries
docs/                 design notes
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- The [candle](https://github.com/huggingface/candle) project — the Rust
  ML stack this builds on.
- [mlx](https://github.com/ml-explore/mlx) and `mlx-lm` — the reference
  point for parity comparisons and the kernel-layout patterns that the
  cooperative 8-bit kernel mirrors.
- Google ICLR 2026 [**TurboQuant**](https://arxiv.org/abs/2504.19874)
  paper — the KV-cache compression algorithm at the heart of the codec
  module.
