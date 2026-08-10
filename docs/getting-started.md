# Getting started

Step-by-step walkthrough for setting up `lumen-rs` from a clean clone
and exercising every validated path. Companion to the top-level
[README](../README.md) — the README is the reference; this doc is the
copy-paste recipe.

## 0. Prerequisites

| | |
|---|---|
| Hardware | Apple Silicon (M1/M2/M3/M4) |
| OS | macOS 14+ |
| Rust | `rustup` with stable 1.85+ (edition 2024) |
| Python | 3.10+ (only needed for fixture download or mlx-lm parity) |
| Disk | ~50 GB for full Gemma 4 + Qwen3.6 checkpoints + `target/` |
| Memory | Embedding ≈ 1 GB · Gemma 4 26B-A4B 4-bit ≈ 22 GB |

## 1. Clone the repo + candle fork

`Cargo.toml` references a sibling `candle/` directory carrying our
single-line `clear_kv_cache` patch (see [DEPENDENCIES.md](../DEPENDENCIES.md)).

```bash
mkdir -p ~/your-projects && cd ~/your-projects/

# This repo
git clone https://github.com/rabbitson87/lumen-rs.git

# Sibling candle checkout — must be at ../candle/ relative to lumen-rs/
git clone https://github.com/huggingface/candle.git
# Apply the `pub fn clear_kv_cache` patch (see DEPENDENCIES.md for the diff)
```

Final layout:

```
your-projects/
  ├── candle/         (patched fork)
  └── lumen-rs/       (this repo)
```

## 2. Build

The default build covers the embedding endpoint + `lumen-server`.

```bash
cd ~/your-projects/lumen-rs

cargo build --release                              # embedding only
cargo build --release --features mlx-native        # + Gemma 4 chat
```

Add Qwen3.6 (opt-in) when you also want the Candle-based MoE backend:

```bash
cargo build --release -p lumen-server
```

The first build downloads dependencies and compiles ~250 crates — allow
5–10 minutes on a clean machine. Subsequent builds are incremental.

### Verify

```bash
cargo test -p lumen-mlx
# expected: 7 passed; 0 failed
```

## 3. Download models

`lumen-rs` does not bundle any weights. Set env vars to local
checkpoint directories.

### Qwen3-Embedding-0.6B (MLX 8-bit)

```bash
huggingface-cli download \
  mlx-community/Qwen3-Embedding-0.6B-8bit-mlx \
  --local-dir ~/models/qwen3-embedding-0.6b-8bit

export EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit
```

### Gemma 4 26B-A4B (MLX 4-bit)

```bash
huggingface-cli download \
  mlx-community/gemma-4-26b-a4b-mlx-4bit \
  --local-dir ~/models/gemma-4-26b-a4b-mlx-4bit

export MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit
```

On 24 GB machines, use the 3-bit variant
(`mlx-community/gemma-4-26b-a4b-mlx-3bit`, ~16 GB resident).

### Qwen3.6 35B-A3B mxfp4 (optional, opt-in)

```bash
huggingface-cli download \
  mlx-community/Qwen3.6-35B-A3B-mxfp4 \
  --local-dir ~/models/qwen3.6-35b-a3b-mxfp4

export LUMEN_QWEN35_SHARDS=~/models/qwen3.6-35b-a3b-mxfp4
```

## 4. Run the bundled examples

### Embedding smoke

Loads model, embeds 3 KR/EN phrases, prints timings + cosines.

```bash
cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity
```

Expected (M3 Max):

```
[smoke] embed b=3 over 5 iters: median=19.35ms min=18.72ms max=22.35ms
[smoke] OK: semantic ordering preserved
```

### Embedding quality (25-item KR/EN corpus)

```bash
cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity
```

Expected:

```
[quality] P@1 = 0.960   P@3 = 0.880   MRR = 0.980
```

### Compare qmv_fast vs naive 8-bit kernel

```bash
# Cooperative simdgroup (default):
cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity

# Naive 1-thread/output (forced):
LUMEN_EMBEDDING_BATCH_ROWS=1 \
  cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity
```

You should see ~19 ms vs ~35 ms — proof the kernel optimization
preserves quality (identical cosine ordering) while halving latency.

### Gemma 4 native E2E bench

```bash
cargo run --release --features mlx-native \
  -p lumen-mlx --example bench_gemma4_native_e2e -- \
  PROMPT_LEN=4096 STEPS=32 WARMUP=8
```

Expected (M3 Max): ~18.8 ms/step decode with the custom flash-attn
primitive (vs ~19.9 ms with mlx default sdpa).

### Qwen3.6 MLX bench

```bash
MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
PROMPT_LEN=2048 STEPS=32 WARMUP=8 \
  cargo run --release --features mlx-native \
    -p lumen-mlx --example bench_mlx_e2e
```

Expected (M3 Max): ~14.85 ms/step p50, 67.3 tok/s.

## 5. Run the HTTP server

### Embedding + Gemma 4 (recommended)

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit \
  cargo run --release --features mlx-native --bin lumen-server
```

Default listen address: `127.0.0.1:8080`. Override with `HOST` / `PORT`.

### Embedding only

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release --bin lumen-server
```

`/v1/chat/completions` returns 503 when no chat model is configured.

### + Qwen3.6 chat (opt-in)

```bash
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit \
  cargo run --release --bin lumen-server
```

### Smoke-test the endpoints

```bash
# Embeddings
curl -s localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b","input":["hello","안녕"]}' \
  | jq '.data[0].embedding | length'   # → 1024

# Chat
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b",
    "messages": [{"role":"user","content":"Capital of South Korea?"}],
    "max_tokens": 32
  }' | jq -r '.choices[0].message.content'
```

### Image input (optional)

Gemma 4 and Qwen 3.6 both ship an image tower, but neither is loaded
unless you ask for it — it costs ~1.1 GB (Gemma 4) or ~0.9 GB (Qwen 3.6)
on top of the text weights. Restart the server with:

```bash
LUMEN_VISION=1 \
LUMEN_VISION_MAX_SOFT_TOKENS=140 \
MODEL_ID=~/models/mlx-community--gemma-4-26b-a4b-it-4bit \
  cargo run --release --features mlx-native --bin lumen-server
```

`LUMEN_VISION_MAX_SOFT_TOKENS=140` halves the patch grid versus the
checkpoint default of 280, which keeps peak memory near the text-only
footprint on a 36 GB machine.

Qwen 3.6 is the same flag, with a token budget instead of a soft-token
one (`LUMEN_VISION_MAX_IMAGE_TOKENS`, default 1024):

```bash
LUMEN_VISION=1 \
LUMEN_VISION_MAX_IMAGE_TOKENS=512 \
MODEL_ID=~/models/Qwen3.6-27B-MTPLX-Speed \
  cargo run --release --features mlx-native --bin lumen-server
```

Either way, the request looks the same:

```bash
B64=$(base64 -i some-image.png)
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"gemma-4-26b-a4b\",
    \"max_tokens\": 128,
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"Describe this image in one sentence.\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}
    ]}]
  }" | jq -r '.choices[0].message.content'
```

Only `data:` URLs are accepted — the server will not fetch remote ones.
If the reply ignores the image, check the startup log: checkpoints that
were requantized without their `vision_tower.*` weights print a warning
and leave image input disabled. There is also a CLI path that skips HTTP
entirely:

```bash
LUMEN_VISION=1 MODEL_ID=~/models/mlx-community--gemma-4-26b-a4b-it-4bit \
  cargo run --release --features mlx-native -p lumen-mlx \
  --example gemma4_vision_describe -- some-image.png "What is this?"
```

## 7. Common A/B switches

| Env | Default | Effect |
|---|---|---|
| `LUMEN_EMBEDDING_BATCH_ROWS=1` | off | Force naive 8-bit GEMM (embedding path) |
| `KESTREL_GEMMA4_CUSTOM_FLASH_ATTN=0` | on | Disable custom flash-attn primitive |
| `KESTREL_GEMMA4_PREFILL_SYNC=0` | on | Skip explicit eval-sync after prefill |
| `LUMEN_DISABLE_FLASH_ATTN=1` | off | Disable Qwen3.6 flash-attn |
| `LUMEN_DISABLE_RESIDUAL_FUSION=1` | off | Disable Qwen3.6 residual+RMSNorm fusion |
| `LUMEN_DISABLE_INPUT_RMSNORM_FUSION=1` | off | Disable Qwen3.6 input-norm fusion |
| `LUMEN_DISABLE_DENSE_MLP_RESIDUAL_FUSION=1` | off | Disable Qwen3.6 dense-MLP residual fusion |
| `LUMEN_DISABLE_MOE_GATE_UP_SILU_MUL_FUSION=1` | off | Disable Qwen3.6 MoE gate/up/silu/mul fusion |
| `LUMEN_DISABLE_MOE_WSUM_FUSION=1` | off | Disable Qwen3.6 MoE weighted-sum fusion |
| `LUMEN_MLX_BACKEND=native\|pyo3\|subprocess` | **native** | Picks the mlx runner. `native` is the Apple-silicon-optimized mlx-rs path |

## 8. Troubleshooting

See the [README's Troubleshooting section](../README.md#troubleshooting)
for the standard pitfalls (missing candle fork, missing model env vars,
slow embedding latency, OOM on Gemma 4). The rest of this section
documents issues specific to the validation workflow.

### Thermal throttling skews A/B numbers

The mean step-latency reported by the bench is sensitive to thermal
state. After a hot run, the mean can drift +50 % even though the
algorithmic comparison is unchanged. Use `p50` (printed alongside
`mean`) as the apples-to-apples metric; let the machine cool for 60 s
between A/B runs.

### `hf upload` 403 Forbidden

The fixtures dataset is under the `hsng95` namespace by default. If you
clone this repo and want to publish your own fixture builds, replace
(`DEFAULT_REPO`) and use a token with **Write + Create repos** scope.
Fine-grained tokens need both checks; classic "Write" tokens cover
both implicitly.

### Slow `cargo build` first time

The default build compiles MLX from source (cmake + the Metal shader
compiler), which dominates the first build. Use `cargo build --release`
with `CARGO_BUILD_JOBS`
matched to your performance cores to avoid scheduler thrash on small
M-series machines.

## 9. What next?

- The TurboQuant codec (`lumen-core`) is hardware-agnostic — a CUDA
  backend would only need a new dispatcher.
- PagedAttention scaffolding lives in `paged-attention/`. Not yet wired
  to the server; an experimental kernel-level test exists behind
  `--features legacy-tests`.
- Spec-decode + MTP draft heads are partially implemented for the
  Qwen3.6 path; see `LUMEN_SPEC=mtp` and `LUMEN_QWEN35_HF_ORIGINAL`.

For day-to-day questions, the source-level docstrings under
`crates/*/src/` are the authoritative answer — most non-obvious
behavior is annotated where it lives.
