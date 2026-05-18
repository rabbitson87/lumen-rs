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
  - `Pin candle + mlx-rs + mlx-c + mlx forks via git URL + commit SHA`
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
# 1. Default build (embedding + qwen3.6 candle path + lumen-server)
cargo check --workspace --release

# 2. Opt-in feature builds
cargo check --workspace --features lumen-server/qwen3_5_moe --release
cargo check --workspace --features lumen-server/mlx-native --release

# 3. Bit-parity tests for the affine8 kernel (always-on validation)
cargo test -p lumen-metal --release --test affine8_parity
# Expected: 7 passed; 0 failed

# 4. Personal-path leak scan
git grep "/Users/sonheesung"
# Expected: empty output
```

Anything that prints `error[E...]` blocks the commit. Compile warnings
are acceptable (the workspace has ~50 known warnings; the perf
A/B path is the source of most).

---

## 3. Fork dependency management

`lumen-rs` consumes four upstream forks pinned to specific commit SHAs.
Each fork lives under `github.com/rabbitson87/<fork>` with a branch
named `lumen-rs-patches`.

| Cargo dep | Fork branch | URL |
|---|---|---|
| `candle-*` | `rabbitson87/candle/lumen-rs-patches` | https://github.com/rabbitson87/candle |
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
cargo check --workspace --release         # confirm rebuild succeeds
git commit -am "deps: bump <fork> to <short-sha>"
git push origin main
```

### Local fork-edit workflow (uncomment `[patch]` overrides)

The top-level `Cargo.toml` has a commented `[patch]` block at the
bottom. Uncomment it to redirect git deps to sibling local clones
(`../candle/`, `../../../mlx-rs/`). For mlx-c / mlx, use the
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
  !scripts/fetch_fixtures.py       allowlisted: clone-er-friendly utility
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
| `metal` | `lumen-model` | Candle Metal backend | ✅ on |
| `turboquant-gpu` | `lumen-model`, `lumen-server` | GPU-resident affine 4/8-bit dispatch | ✅ on |
| `paged-kv` | `lumen-model` | PagedAttention KV cache scaffolding | ✅ on |
| `qwen3_5_moe` | `lumen-model`, `lumen-server` | Qwen3.5 / Qwen3.6 MoE Candle backend (opt-in) | ❌ off |
| `mlx-native` | `lumen-mlx`, `lumen-server` | Native MLX Gemma 4 26B-A4B path | ❌ off |
| `mlx-pyo3` | `lumen-mlx` | PyO3 subprocess fallback (development only) | ❌ off |
| `turboquant` | `lumen-model`, `lumen-server` | Candle-side Gemma 4 E4B turboquant path (requires `[patch]` override — uses workspace-relative deps) | ❌ off |
| `legacy-tests` | `lumen-metal`, `paged-attention` | Re-enable broken integration tests that drifted out of sync with current APIs | ❌ off |

When adding new features:

- **Off by default** unless validated end-to-end on the target hardware.
- **Cargo.toml `[[test]] / [[example]]` `required-features`** when an
  integration test / example relies on the feature; this prevents
  default-test runs from breaking.
- **README + `docs/getting-started.md`** entry in the feature table.

---

## 7. HuggingFace fixture pipeline

| | |
|---|---|
| Dataset repo | `hsng95/lumen-rs-fixtures` |
| Files | `layer0_moe_weights.safetensors` (3.2 GB), `layer0_linear_attn_weights.safetensors` (135 MB), `layer3_self_attn_weights.safetensors` (109 MB) |
| Download | `python scripts/fetch_fixtures.py` (uses `DEFAULT_REPO` constant; override with `LUMEN_FIXTURES_REPO`) |
| Upload | `hf upload hsng95/lumen-rs-fixtures <file> --repo-type dataset` (needs `hf auth login` with **Write** scope) |
| zsh gotcha | Don't use `\` line continuation when chaining `hf upload` — zsh fights it. Run each upload as a single line. |

If you add new fixtures, edit the `FIXTURES` list in
`scripts/fetch_fixtures.py` *and* upload to the dataset in the same
PR; otherwise clone-ers' `fetch_fixtures.py` runs error out.

---

## 8. Performance reference numbers (M3 Max, sanity gates)

Use these as the "did I break something?" thresholds. If a perf bench
deviates by more than 30% (p50) without a known cause, investigate
before pushing.

| Workload | Expected | A/B partner |
|---|---|---|
| Embedding b=3, default kernel | ~19 ms/batch | naive (`LUMEN_AFFINE8_NAIVE=1`): ~35 ms |
| Embedding quality eval (25-item KR/EN) | P@1 ≥ 0.95, MRR ≥ 0.97 | — |
| Gemma 4 26B-A4B decode (custom flash-attn) | ~18.8 ms/step | mlx default sdpa (`KESTREL_GEMMA4_CUSTOM_FLASH_ATTN=0`): ~19.9 ms |
| Qwen3.6-35B-A3B-mxfp4 N=1 decode | 48 ms/step p50, 20 tok/s | all `LUMEN_DISABLE_*=1`: ~66 ms p50 |
| Qwen3.6 N=1 → N=2 CB | aggregate +17% | per-seq −41% (known MoE limitation) |

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
| `/v1/embeddings` (Qwen3-Embedding-0.6B MLX 8-bit) | ✅ validated | `embedding_smoke` runs ~19 ms; `embedding_quality` P@1 = 0.960 |
| `/v1/chat/completions` (Gemma 4 26B-A4B MLX 4-bit, mlx-native) | ✅ validated | `bench_gemma4_native_e2e` ~18.8 ms/step, matches mlx-lm within 1 ms |
| `/v1/chat/completions` (Qwen3.6-35B-A3B-mxfp4, qwen3_5_moe) | ✅ validated | `bench_cb_qwen35` p50 48 ms / 20 tok/s |
| `/v1/chat/completions` (Qwen3.6-27B-4bit dense, qwen3_5_moe) | ⚠ partially validated | Same code path; only the 35B-A3B variant has bench numbers |
| GGUF Gemma | ⚠ exploratory | No active CI |
| Candle Qwen legacy | ⚠ exploratory | Pre-mlx-native; left in place |
| PagedAttention scheduler | ❌ WIP | Scaffolding only; integration tests behind `legacy-tests` |

---

## 10. Common operation cheatsheet

```bash
# Smoke before push (full matrix)
cargo check --workspace --release
cargo check --workspace --features lumen-server/qwen3_5_moe --release
cargo check --workspace --features lumen-server/mlx-native --release
cargo test -p lumen-metal --release --test affine8_parity
git grep "/Users/sonheesung"     # must be empty

# Quick embedding A/B
cargo run --release -p lumen-model --example embedding_smoke
LUMEN_AFFINE8_NAIVE=1 cargo run --release -p lumen-model --example embedding_smoke

# Gemma 4 native bench
MODEL_ID=~/models/gemma-4-26b-a4b-mlx-4bit \
PROMPT_LEN=4096 STEPS=32 WARMUP=8 \
  cargo run --release --features mlx-native \
  -p lumen-mlx --example bench_gemma4_native_e2e

# Qwen3.6 CB bench
SHARDS=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/*/ | head -1)
LUMEN_QWEN35_SHARDS="$SHARDS" \
MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
N=2 DECODE_STEPS=32 WARMUP=4 PROMPT_LEN=128 \
  cargo run --release --features qwen3_5_moe \
  -p lumen-model --example bench_cb_qwen35

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

## Update policy for this document

Treat this file as a living maintainer guide. When you (the maintainer
or an agent acting on their behalf) introduce a new pattern that
should be reused — a new feature flag, a new fork, a new perf
benchmark, a new validation step — **edit this file in the same
commit**. The goal is that the next session can drop in cold and find
the right defaults without re-discovering them.
