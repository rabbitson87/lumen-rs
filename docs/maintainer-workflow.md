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
| PagedAttention | ❌ parked, **measured** | `crates/paged-attention` stays excluded from the workspace. `kv_concurrency_ab` (M3 Max, Qwen3.5-9B, N=1/2/4/8, three length profiles) puts the reclaimable block-rounding slack at 72 MB / 66 MB / 35 MB for short / mixed / long prompts — **0.35–0.91% of process memory**. 40–63% of per-sequence resident memory is linear-attention SSM state that paging cannot compact, and the real ceiling is the prefill `[1, prompt_len, vocab]` logits tensor (11.5 GB at N=8 long) which it also cannot touch. See the crate README for the full table and the two larger levers it surfaced |

The Candle rows (Candle continuous batching, GGUF Gemma, Candle Qwen legacy)
are gone with the backend. GGUF has no MLX equivalent, so that capability was
dropped rather than ported; it had already been unreachable in a default build,
since `mlx-native` short-circuits backend selection before the GGUF check.

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
