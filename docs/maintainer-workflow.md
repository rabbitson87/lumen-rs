# Maintainer workflow

This document captures the conventions and operational patterns that the
`main` branch is currently maintained against. Future commits (whether
authored by the maintainer or by an AI agent operating on their behalf)
should follow these defaults unless the user explicitly directs
otherwise.

For agent sessions: read this first. The patterns below override any
generic defaults from `CLAUDE.md` for this repo.

---

## 1. Commit message style

Match the existing log. Each commit message is:

- **Title line** (≤ 72 chars). Verb-led, descriptive. Examples that
  matched the user's intent:
  - `Initial public release (alpha / experimental)`
  - `Add Qwen3.5/3.6 MoE backend (opt-in via --features qwen3_5_moe)`
  - `Add Qwen3.6-35B-A3B-mxfp4 A/B numbers + HF Hub fixture pipeline`
  - `fix: HF fixtures namespace rabbitson87 -> hsng95`
  - `docs: add docs/getting-started.md step-by-step walkthrough`
  - `Pin mlx-rs + mlx-c + mlx forks via git URL + commit SHA`
- **Blank line**, then a body that explains *why* in plain prose.
- **Bulleted sub-sections** with `*` markers when the change touches
  multiple subsystems.
- **Concrete artifacts**: file paths, function names, SHAs, env vars.
  Avoid hand-wavy summaries.
- **Use HEREDOC** for `git commit -m "$(cat <<'EOF' ... EOF)"` to
  preserve formatting (the global `CLAUDE.md` example applies).

Conventional-commit prefixes (`fix:`, `docs:`, `feat:`) are OK but not
required. The user has used both styles; pick whichever fits the
change scope.

---

## 2. Pre-commit / pre-push validation gates

Before staging anything, run these — in order, fail fast:

```bash
# 1. The suite, with the features and serialization that make it
#    representative. See xtask/src/test_all.rs for why a plain
#    `cargo test --workspace` is not.
cargo xtask test
# Expected: 0 failed. Note what is *ignored* — that count is how many
# tests did not run, and it is meant to be looked at, not scrolled past.

# 2. Every regression guard still catches its defect.
cargo xtask red-green
# Expected: all PASS. A VACUOUS or ALREADY-RED verdict fails the run.

# 3. The GPU-free build compiles and its tests run in seconds. This is the
#    configuration that silently rotted before (`qwen3_5_mtp` lost its
#    feature gate and nothing noticed for months).
cargo test -p lumen-mlx
cargo test -p lumen-server --no-default-features --lib

# 4. Personal-path leak scan
git grep "/Users/sonheesung"
# Expected: empty output
```

Anything that prints `error[E...]` blocks the commit. Compile warnings
are acceptable (the workspace has known warnings; the perf A/B path is
the source of most).

---

## 3. Fork dependency management

`lumen-rs` consumes three upstream forks pinned to specific commit SHAs.
Each fork lives under `github.com/rabbitson87/<fork>` with a branch
named `lumen-rs-patches`.

The `candle-*` fork was dropped when the Candle backend was removed — a
clean clone no longer needs a sibling `../candle` checkout to build.

| Cargo dep | Fork branch | URL |
|---|---|---|
| `mlx-rs`, `mlx-sys` | `rabbitson87/mlx-rs/lumen-rs-patches` | https://github.com/rabbitson87/mlx-rs |
| mlx-c submodule | `rabbitson87/mlx-c/lumen-rs-patches` | https://github.com/rabbitson87/mlx-c |
| mlx core (FetchContent in mlx-c) | `rabbitson87/mlx/lumen-rs-patches` | https://github.com/rabbitson87/mlx |

Each branch is a **single squashed commit on top of an upstream
baseline** so rebases stay atomic.

### Bumping a fork SHA (when upstream has new commits to integrate)

```bash
cd ~/Documents/GitHub/<fork>
git fetch upstream
git checkout lumen-rs-patches
git rebase upstream/main          # or upstream/<tag>
# Resolve any conflicts (rare, since we only carry one squashed commit)
git push --force-with-lease <remote> lumen-rs-patches
git rev-parse HEAD                # copy the new SHA

cd ~/Documents/GitHub/lumen-rs
# Update the `rev = "..."` field in Cargo.toml (and lumen-mlx/Cargo.toml
# for mlx-rs / mlx-sys). For mlx-c bumps, the SHA lives in mlx-rs's
# submodule pin; for mlx core bumps, in mlx-c's CMakeLists.txt
# FetchContent_Declare.
cargo xtask test                          # confirm rebuild + suite
git commit -am "deps: bump <fork> to <short-sha>"
git push origin main
```

### Local fork-edit workflow (uncomment `[patch]` overrides)

The top-level `Cargo.toml` has a commented `[patch]` block at the
bottom. Uncomment it to redirect the git dep to a sibling local clone
(`../../../mlx-rs/`). For mlx-c / mlx, use the
`MLX_LOCAL_SOURCE_DIR` env var on the mlx-c CMakeLists.txt FetchContent
call (set it before `cargo build`).

**Never commit the uncommented `[patch]` block** — public clones break
because they don't have the sibling repos. Use a local
`Cargo.toml.user` symlink or `git update-index --skip-worktree` if you
need persistent overrides without polluting the tree.

---

## 4. Push policy

- `main` is the single long-running branch.
- Force-push with `--force-with-lease` is OK on `main` because the
  user is the sole maintainer. **Always with-lease**, never `--force`
  alone.
- Amending the latest commit and force-pushing is the preferred way to
  fix small issues spotted right after a push (typo in commit message,
  missing file, etc.). For substantive follow-ups, prefer a new
  commit.
- Don't rewrite published commits older than `HEAD`. If you find an
  issue in an older commit, fix it forward.

---

## 5. Personal-hygiene scrubbing

Everything in the following list is **gitignored** and must stay that
way. New files matching these patterns should never be added to a
commit:

```
.ai/                               personal AI workflow state
.claude/                           Claude Code config
.lean-ctx/                         lean-ctx context cache
.outline/                          investigation artifacts (Instruments traces, etc.)
.venv/                             Python venv
AGENTS.md, AI-RULES.md             personal agent prompts
LEAN-CTX.md, ai-system-install-*.md
claude.md                          121KB personal config
scaffold.mjs                       personal AI installer
scripts/*                          personal benchmarking scripts
worker-hang-investigation.html     debug artifact
lumen-rs                           self-referential symlink (don't recreate)
```

Plus content scrubs that apply to all `*.rs`, `*.md`, `*.py`, `*.sh`,
`*.json` files in committed history:

- **No `/Users/sonheesung` paths** — replace with `/path/to/...`,
  `~/models/...`, or env-var-required (no default).
- **No personal email** (`hsng95@gmail.com`) except where it's
  intentional attribution (the LICENSE copyright line).
- **No `claude.md` / `.ai/memory` references** in source comments.
- **Phase X.Y prefix cleanup** — strip `Phase 1.8 M4 — ...`,
  `Phase 17.B: ...`, `CB Phase 2: ...` etc. at the start of comments.
  Keep mid-prose Phase references when they provide context. See the
  Python regex in the session's path-scrub commit for the patterns.

When in doubt, run:

```bash
git grep -i "sonheesung\|/Users/\|claude\.md\|hsng95"
git grep -E "^\s*//[!/]?\s*Phase\s+"
```

The first should return zero hits (modulo LICENSE). The second
should return only mid-prose references (Phase X mid-sentence is OK).

---

## 6. Feature flag conventions

| Feature | Crate | Purpose | Default? |
|---|---|---|---|
| `mlx-native` | `lumen-mlx`, `lumen-diffusion`, `lumen-server` | Native MLX runner (Gemma 4, Qwen 2.5/3.5/3.6 dense + MoE) plus the FLUX.2-dev diffusion backend. The only backend. | ✅ **on** (default) |
| `mlx-native-metal` | `lumen-mlx`, `lumen-diffusion` | `mlx-native` + `mlx-rs/metal` | ❌ off |
| `mlx-pyo3` | `lumen-mlx` | PyO3 subprocess fallback (development only) | ❌ off |

The Candle-era features (`metal`, `turboquant-gpu`, `paged-kv`,
`qwen3_5_moe`, `legacy-candle`, `turboquant`, `mpsgraph`, `legacy-tests`)
were removed with the Candle backend.

When adding new features:

- **Off by default** unless validated end-to-end on the target hardware.
- **Cargo.toml `[[test]] / [[example]]` `required-features`** when an
  integration test / example relies on the feature; this prevents
  default-test runs from breaking.
- **README + `docs/getting-started.md`** entry in the feature table.

---

## 8. Performance reference numbers (M3 Max, sanity gates)

Use these as the "did I break something?" thresholds. If a perf bench
deviates by more than 30% (p50) without a known cause, investigate
before pushing.

| Workload | Expected | A/B partner |
|---|---|---|
| Embedding, 25 texts warm | ~55 ms total, 2.20 ms/item | unbatched (`LUMEN_EMBEDDING_BATCH_ROWS=1`): 220 ms, 8.80 ms/item |
| Embedding quality eval (25-item KR/EN) | P@1 ≥ 0.95, MRR ≥ 0.97 | vs the committed Candle reference: per-item cosine ≥ 0.99 |
| Gemma 4 26B-A4B decode (custom flash-attn) | ~18.8 ms/step | mlx default sdpa (`KESTREL_GEMMA4_CUSTOM_FLASH_ATTN=0`): ~19.9 ms |
| Qwen3.6-35B-A3B-mxfp4 N=1 decode | 13.9 ms/step p50, **71.6 tok/s** | — |
| Qwen3.6-35B-A3B-mxfp4 PROMPT_LEN=2048 decode | 14.85 ms/step p50, **67.3 tok/s** | — |

The Candle A/B columns are gone with the backend. Their last recorded values,
for the record: Candle N=1 was 22.0 ms / 45.5 tok/s, and at PROMPT_LEN=2048 it
was 486 ms / 2.0 tok/s — its SDPA did not scale to long KV, which is most of
why MLX became the only path.

**Always report `p50` for thermal-sensitive measurements** — `mean` is
noisy after a few hot runs. The benches print both.

When running A/B benches, insert a 60 s cooldown between back-to-back
runs. The user's M3 Max throttles aggressively after ~3 consecutive
high-load runs.

---

## 9. Validated paths (what works end-to-end)

Update this table whenever a path graduates from "WIP" to "validated".

| Path | Status | Validation evidence |
|---|---|---|
| `/v1/embeddings` (Qwen3-Embedding-0.6B MLX 8-bit) | ✅ validated | `embedding_parity` — worst per-item cosine 0.9988 vs the captured Candle reference, P@1 0.960 / MRR 0.980 identical, 2.20 ms/item warm (4.6× the Candle path it replaced) |
| `/v1/chat/completions` (Gemma 4 26B-A4B MLX 4-bit) | ✅ validated | `bench_gemma4_native_e2e` ~18.8 ms/step, matches mlx-lm within 1 ms |
| `/v1/chat/completions` (Qwen3.6-35B-A3B-mxfp4) | ✅ validated | `bench_mlx_e2e` p50 13.94 ms / **71.6 tok/s** (PROMPT_LEN=8), 14.85 ms / 67.3 tok/s (PROMPT_LEN=2048) |
| `/v1/chat/completions` (Qwen3.6-27B-4bit dense) | ⚠ partially validated | Same code path; only the 35B-A3B variant has bench numbers |
| `/v1/images/generations` (FLUX.2-dev) | ✅ validated | 512² generations; see the diffusion port notes |
| PagedAttention | ❌ **removed**, with the measurement on record | Deleted, not parked — see below |

The Candle rows (Candle continuous batching, GGUF Gemma, Candle Qwen legacy)
are gone with the backend. GGUF has no MLX equivalent, so that capability was
dropped rather than ported; it had already been unreachable in a default build,
since `mlx-native` short-circuits backend selection before the GGUF check.

### PagedAttention: measured, then deleted

`crates/paged-attention` was parked by task 006 and **deleted** after task 007
measured what it would buy. The measurement is the reason; it is recorded here
rather than in the crate, because the crate is gone.

Reproduce with `cargo run --release -p lumen-mlx --features mlx-native
--example kv_concurrency_ab`. On M3 Max / 36 GiB against Qwen3.5-9B (8
full-attention layers of 32), at N = 1/2/4/8 over three prompt-length profiles,
resident memory fits `~53 MB x N + ~67 KB x allocated_slots` with no intercept
at R² ≥ 0.998, the three profiles agreeing to ~10%. Replacing the 256-token
`KV_CACHE_STEP` blocks with 16-token paged blocks would reclaim:

| prompt profile (N=8) | resident | reclaimable | % of process |
|---|---|---|---|
| short, 120–480 tok | 661.8 MB | 72.2 MB | **0.91%** |
| mixed, 200–3000 tok | 1,064.8 MB | 65.8 MB | **0.79%** |
| long, 2000–8000 tok | 2,960.3 MB | 35.4 MB | **0.35%** |

Short turns are paging's best case and it is still 72 MB against a 7.9 GB
process. Two structural reasons, neither visible without measuring: **40–63% of
per-sequence residency is linear-attention conv/SSM state**, which is
length-independent and which paging — a full-attention KV technique — cannot
compact; and **the binding constraint is prefill, not KV** (11.5 GB peak at N=8
long against a 3.1 GB decode peak). There was no throughput gap either (batched
decode already scales ~2.4× from N=1 to N=8) and no leak (0.0 MB residual after
`remove_seq` + `clear_cache` at every width).

### The prefill peak is a tuning knob — with a real price at long prompts

Prefill is **already chunked** — `qwen35_prefill_chunk()`, default 2048 tokens,
with an always-chunk invariant — and `last_only` already collapses the lm_head
to a single row, so no `[1, prompt_len, vocab]` tensor is ever materialized. The
prefill peak is per-chunk activations, and it tracks the chunk rather than the
prompt.

**Output is chunk-invariant.** `forward_chunked`'s equivalence argument (RoPE
and the causal sentinel key off `cache.offset()`; linear-attn layers carry conv/
SSM state through the cache) is now tested, not just reasoned:
`examples/prefill_chunk_equivalence.rs` prefills one prompt at several chunk
sizes against the same loaded weights — `qwen35_prefill_chunk()` re-reads the
env on every call, so the A/B shares a process — and compares greedy token ids.
Bit-identical at 256/512/1024/2048 across every run, at 8K and 20K prompts.

**The chunk is a memory/latency trade, and the price is not flat.** Measured on
Qwen3.5-9B / M3 Max, `--gen 8..32`, active memory sampled immediately after
prefill returns:

| prompt | chunk | prefill | memory after prefill |
|---|---|---|---|
| 8,007 tok | 2048 | baseline | 2,055 MB |
| | 1024 | no detectable Δ | 1,250 MB |
| | 512 | no detectable Δ | 849 MB |
| | 256 | no detectable Δ | 646 MB |
| 20,000 tok | 2048 | baseline (52–59 s) | 2,556–2,621 MB |
| | 1024 | **+9 to +17%** | 1,750–1,816 MB |
| | 512 | **+54 to +118%** | 1,348–1,414 MB |

At 8K the time deltas over five runs average ~0 against a ±11% run-to-run noise
floor, so there is nothing to detect. At 20K the cost is unmistakable and
reproduces with the sweep order reversed (512 first: 113 s, 2048 second: 52 s),
so it is not a thermal or ordering artifact. The cost tracks chunk **count** —
each chunk is `eval`'d before the next, and 40 serialization points cost more
than 10.

**So the default stays at 2048.** Lowering it globally would tax exactly the
long agentic prompts that matter most. `LUMEN_QWEN35_PREFILL_CHUNK` remains the
escape hatch for a memory-tight machine that would rather pay latency, and the
table above is the exchange rate.

### bf16 KV storage — measured, shipped default-off (`LUMEN_MLX_KV_BF16`)

The largest memory lever 007 surfaced is now implemented and measured.
`examples/kv_bf16_ab.rs` flips an atomic (`set_kv_store_bf16`) so both
conditions run against one set of loaded weights in one process. Four operating
points, Qwen3.5-9B / M3 Max, 32 greedy tokens each:

| N | prompt | KV f32 | KV bf16 | saved | KB/slot | 1st-token | sequence | decode Δ |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,002 | 190.7 MB | 123.5 MB | 35.3% | 32.8 | 1/1 | 100% | +2.9% |
| 1 | 8,004 | 593.6 MB | 323.1 MB | 45.6% | 33.0 | 1/1 | 100% | +3.0% |
| 8 | 803 | 980.6 MB | 696.3 MB | 29.0% | 34.7 | 8/8 | 97% | +4.1% |
| 8 | 2,519 | 1,802.7 MB | 1,123.3 MB | 37.7% | 32.8 | 8/8 | 100% | +1.6% |

The per-slot saving lands at 32.8–34.7 KB against a physical prediction of
33.7 KB (f32 minus bf16 per-slot cost) on all four shapes. The headline
percentage varies only because total resident also carries the ~53 MB/sequence
linear-attention state, which no KV dtype touches. **Decode gets faster**, +1.6
to +4.1%, since attention reads half the bytes.

Two controls, because the first run of this sweep produced a nonsense result
and it is worth knowing why:

- **Memory readings need settling.** One `clear_cache()` is not enough — the
  same condition at the same point reported either 190.6 MB or 317.9 MB, a
  fixed ~127 MB quantum present or absent. Pairing a clean reading of one
  condition with a contaminated reading of the other turned a real 32.8 KB/slot
  saving into an apparent 29.4 KB/slot *regression*. The harness now loops
  `clear_cache` until two readings agree; an f32-vs-f32 control puts residual
  measurement noise at 1.9 KB/slot, ~17× below the signal.
- **`--control` runs both conditions in f32**, so output mismatches can be
  attributed. At the one point showing 97% sequence match the control is
  272/272, so MLX is deterministic there and that 3% is genuinely bf16
  rounding rather than scheduling noise.

#### Quality: 6,300 teacher-forced positions, two models

The four-point A/B above measured quality with free-running greedy decode over
random filler prompts, which is weak twice over — filler leaves the next-token
distribution flat, so argmax flips more readily than on real text, and once one
argmax flips every later token is a *different* continuation rather than a
worse one, so the match rate stops meaning anything.

`examples/kv_bf16_quality.rs` fixes both. Six realistic seed prompts (code,
prose EN/KO, reasoning, technical KO, structured JSON) are each extended by the
model's own greedy continuation — in-distribution by construction — and then
both conditions are **teacher-forced** over the identical token sequence, with
per-position argmax compared via `forward_probe`. One sequence yields hundreds
of independent comparisons instead of one cascading trajectory.

| model | positions | f32-vs-f32 control | bf16 agreement | flips |
|---|---|---|---|---|
| Qwen3.5-9B, 6-bit | 4,050 | **100.000%** | 99.827% | 7 |
| Qwen3.6-27B, 4-bit | 2,250 | **100.000%** | 99.733% | 6 |

The control is exactly 100% on both, so every flip is real rather than
scheduling noise. Agreement is flat across context depth — 99.90 / 99.75 /
99.90 on the 9B and 99.65 / 99.73 / 99.82 on the 27B for shallow / mid / deep
thirds — so bf16 error does **not** accumulate as the cache fills, which was
the failure mode that would have ruled out a default flip outright.

**Every flip was a statistical tie.** `ProbeRows` now carries the per-position
`top1 - top2` logit gap, which is what separates "broke a tie differently" from
"changed a confident prediction":

| model | gap at flipped positions | all positions (p1 / p50 / p90) | largest flip's percentile |
|---|---|---|---|
| 9B | 0.0009 – 0.0132 | 0.026 / 2.93 / 13.08 | **0.5th** |
| 27B | 0.0005 – 0.0688 | 0.050 / 3.85 / 11.86 | **1.5th** |

Against a median gap of 3-4 logits, every disagreement sat below the 1.5th
percentile. bf16 did not change a single prediction the model held with any
confidence; it re-broke ties that were numerically ambiguous already, where the
f32 winner has no claim to being the more correct one.

**Default is still off** — that is a decision, not a gap in the evidence.
Memory, speed and quality all point the same way, but flipping it changes what
every request returns (a 600-token reply would differ from today's roughly a
third of the time, at tied positions), and that is a call to make deliberately.
The measurements needed to make it are above; reproduce with
`kv_bf16_quality --control` then without.

The other lever 007 named — `LUMEN_QWEN35_PREFILL_CHUNK` — is covered above; it
is a knob, not a default change.

**To recover the deleted code**, `git log --diff-filter=D -- crates/paged-attention`
finds the removal commit; the crate is intact in its parent (571 LOC of
scheduler / page-table / sequence logic with no GPU API, plus a 480-line MSL
kernel file). Reopen the question only if the serving profile moves to very high
concurrency on very short turns, or to a non-hybrid model where every layer
holds full-attention KV — re-run `kv_concurrency_ab` first either way.

Six `PAGED_*` env vars went with it (`PAGED_KV`, `PAGED_LAYERS`,
`PAGED_KV_HEADS`, `PAGED_HEAD_DIM_SLIDING`, `PAGED_HEAD_DIM_GLOBAL`,
`PAGED_GLOBAL_EVERY`): the desktop app still emitted them, but nothing had read
them since the Candle backend was removed, so the settings silently did nothing.
The seventh, `PAGED_MAX_BATCH`, was never a paged setting — it is the MLX
scheduler's batch width — and is now `LUMEN_MLX_BATCH_MAX`, with the old name
still accepted as a fallback.

---

## 10. Common operation cheatsheet

```bash
# Smoke before push — see §2
cargo xtask test
cargo xtask red-green
cargo test -p lumen-mlx                              # GPU-free tier, seconds
git grep "/Users/sonheesung"                         # must be empty

# Embedding parity + batching A/B
EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity
LUMEN_EMBEDDING_BATCH_ROWS=1 EMBEDDING_MODEL_ID=~/models/qwen3-embedding-0.6b-8bit \
  cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity

# Gemma 4 native bench
MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit \
PROMPT_LEN=4096 STEPS=32 WARMUP=8 \
  cargo run --release --features mlx-native \
  -p lumen-mlx --example bench_gemma4_native_e2e

# Qwen3.6 MLX e2e bench
MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
PROMPT_LEN=2048 STEPS=32 WARMUP=8 \
  cargo run --release --features mlx-native \
  -p lumen-mlx --example bench_mlx_e2e

# Bump a fork SHA after rebasing
cd ~/Documents/GitHub/<fork>
git fetch upstream
git checkout lumen-rs-patches
git rebase upstream/main
git push --force-with-lease <remote> lumen-rs-patches
NEW_SHA=$(git rev-parse HEAD)
cd ~/Documents/GitHub/lumen-rs
# Edit Cargo.toml `rev = "..."`, run cargo check, commit.
```

---

## 11. Backend metrics convention

The Tauri desktop app's METRICS card (tok/s, ms/step, requests/min) is
fed by parsing `lumen-server`'s **stderr** — not by an in-process metric
channel. A new model backend must emit one line at decode end:

```
[<backend>] <kind> done: <N> tokens in <T_ms>ms (<R> tok/s)
```

Required substrings: `done:` + ` tok/s)`. Use `eprintln!` (not
`tracing::info!` / `println!`). Emit **once per request**, never per
token or per step.

Missing the line → METRICS card stays blank for that model. This was
the actual bug for Gemma 4 (fixed by adding 4 emission sites in
[`crates/lumen-mlx/src/gemma4_backend.rs`](../crates/lumen-mlx/src/gemma4_backend.rs)).

Parser tests live in `parse_tests` mod in
[`crates/lumen-app/src/server.rs`](../crates/lumen-app/src/server.rs).
Add a positive case for any new backend you wire up; run with
`cargo test -p lumen-app --lib parse_tests`.

Full rules + anti-examples + end-to-end flow:
[`docs/backend-metrics-convention.md`](backend-metrics-convention.md).

---

## Update policy for this document

Treat this file as a living maintainer guide. When you (the maintainer
or an agent acting on their behalf) introduce a new pattern that
should be reused — a new feature flag, a new fork, a new perf
benchmark, a new validation step — **edit this file in the same
commit**. The goal is that the next session can drop in cold and find
the right defaults without re-discovering them.
