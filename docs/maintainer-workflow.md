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

One command:

```bash
cargo xtask gate
```

It runs, cheapest-failing-first: a feature-combination check (both the
`default = []` and the `mlx-native` build, `--all-targets`), `fmt`, `clippy`,
the representative test suite, `red-green`, `flags --check`, and the
personal-path/email hygiene greps. Each step prints why it exists when it
fails, so a failure does not send you looking for the rationale.

`--quick` skips the two slow steps (the suite and red-green) for a fast
sanity pass mid-work; it is not a substitute before pushing.

Two deliberate omissions, both explained in `xtask/src/gate.rs`:

- **clippy is blocking**, as of the backlog reaching zero. It was report-only
  while 349 warnings stood, on the reasoning that a gate failing on day one is
  a gate people learn to skip. Report-only turned out to be hiding more than
  warnings: a deny-by-default `approx_constant` **error** had been failing
  `cargo clippy` outright for as long as nothing read its exit code. Four lints
  are allowed workspace-wide in the root `Cargo.toml`, each with its reason
  written next to it; everything else is expected to stay at zero.
- **Soak-scale work is not here.** Fuzzing, the full `Optimization`-flag
  equivalence matrix, coverage, Metal validation and Miri all take minutes to
  hours. They belong to a release: see `docs/release-checklist.md`.

Why a plain `cargo test --workspace` is not enough, in one line each: the
interesting harnesses are feature-gated (green by omission without them), and
parallel libtest threads share one Metal command buffer (intermittent SIGABRT).
`xtask/src/test_all.rs` documents both.

Anything that prints `error[E...]` blocks the commit.

**Regression policy:** no bugfix lands without a `DEFECTS` entry in
`xtask/src/red_green.rs` proving red→green. The rule, and what an entry needs,
is at the end of `docs/release-checklist.md`.

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
| Gemma 4 26B-A4B decode (custom flash-attn) | ~18.8 ms/step | mlx default sdpa (`LUMEN_GEMMA4_CUSTOM_FLASH_ATTN=0`): ~19.9 ms |
| Qwen3.6-35B-A3B-mxfp4 N=1 decode | 13.9 ms/step p50, **71.6 tok/s** | — |
| Qwen3.6-35B-A3B-mxfp4 PROMPT_LEN=2048 decode | 14.85 ms/step p50, **67.3 tok/s** | — |

The Candle A/B columns are gone with the backend. Their last recorded values,
for the record: Candle N=1 was 22.0 ms / 45.5 tok/s, and at PROMPT_LEN=2048 it
was 486 ms / 2.0 tok/s — its SDPA did not scale to long KV, which is most of
why MLX became the only path.

**Re-measured**, load average ~3 (not idle) — which only strengthens a *pass*,
since contention can make a number worse but not better:

Conditions for every row below: `mlx-community--gemma-4-26b-a4b-it-4bit` /
`mlx-community--Qwen3.6-35B-A3B-mxfp4` / `qwen3-embedding-0.6b-8bit`, M3 Max,
2026-08-13, `PROMPT_LEN=8` unless stated.

| Workload | Recorded | Measured | |
|---|---|---|---|
| Embedding, 25 texts warm | ~55 ms, 2.20 ms/item | **57 ms, 2.30 ms/item** | +4% |
| Embedding quality | P@1 ≥ 0.95, MRR ≥ 0.97, cosine ≥ 0.99 | **0.96 / 0.98 / 0.9988** | pass |
| Gemma 4 26B-A4B decode | ~18.8 ms/step | **13.5 and 13.9 ms/step** | ~27% *faster* |
| Qwen3.6-35B-A3B-mxfp4 N=1 | 13.9 ms p50, 71.6 tok/s | **14.20 ms p50, 69.6 tok/s** | +2.2% |
| …same, PROMPT_LEN=2048 | 14.85 ms p50, 67.3 tok/s | **14.62 ms p50, 68.2 tok/s** | 1.5% *faster* |

**The drift is not uniform, and that is the useful part.** Both mxfp4 rows
reproduce within ~2% of numbers recorded on an earlier tree, which says the
harness and the method are sound and that a ±2% band is what "unchanged" looks
like here. Against that, the Gemma 4 row sitting 27% *faster* is not measurement
slop — it is either a real improvement that landed unrecorded, or the original
figure was taken differently (a different prompt length, or before a lever
landed). Worth chasing the next time someone touches that path; a number that
good and that unexplained is as much a smell as a slow one.

### Chasing the Gemma 4 delta: what is established, and what is not

The A/B partner column exists to explain that row, so it was the obvious next
step. Two things came out of it; only one is a measurement.

**Established, from code and config rather than a stopwatch.** The custom
flash-attn kernel can apply to at most **5 of 30 layers**. `layer_types` on
gemma-4-26b-a4b is 25 `sliding_attention` to 5 `full_attention`, and
`use_custom_flash` requires `!use_sdpa_windowed` (plus `S == 1`, `head_dim ==
256`, all-bf16, no explicit mask — see `gemma4_moe.rs`). Whatever the kernel is
worth, it is worth it on a sixth of the attention work, and attention is itself
a small share of a decode step that is dominated by weight reads. That alone
predicts a small A/B delta, and at `PROMPT_LEN=512` the measured delta is
**448 ms vs 449 ms over 31 steps — nothing.**

**Settled, once the measurement stopped being process-per-run.** At
`PROMPT_LEN=8192`, 10 interleaved in-process pairs:

| | median | min |
|---|---|---|
| custom flash-attn ON | 22.54 ms/step | 21.43 |
| OFF (mlx default sdpa) | 23.00 ms/step | 21.50 |

**min-vs-min −0.3%**, median −2.0%, against a 7.0% noise floor. The recorded
5.5% gap (18.8 vs 19.9) does not reproduce: the two paths are indistinguishable
here, which is what the 5-of-30-layers bound predicts.

So the custom kernel is **not** the explanation for the Gemma row's 27% drift.

### Where the drift came from: eliminated, and what is left

The Gemma row was written on **2026-05-19** (`41d3cb4`) and never revised. The
mxfp4 rows next to it *were* revised, later the same day (`bc41ef9`, when
mlx-native became the default runner), which is why one pair reproduces at ±2%
and the other does not. The drift is bookkeeping first and code second.

The comparison is fair: the recorded row states no `PROMPT_LEN`, the bench
default is 8, and 8 is what the re-measurement used. The mxfp4 rows in the same
table *do* state theirs, which is evidence the Gemma row simply took the
default.

**Eliminated, each by measurement or by reading rather than by assumption:**

* *Custom flash-attn* — −0.3% min-vs-min over 10 in-process pairs. Not it.
* *A default flip on the Gemma path* — 47 env gates then, 60 now: 13 new, all
  default-off, and **zero changed defaults**.
* *`LUMEN_MLX_KV_BF16` being flipped by the flags-registry refactor* — it was
  not. The pre-refactor code already read `.unwrap_or(true)`; the migration was
  faithful. (It is on the Gemma path despite being declared in the Qwen module —
  14 uses in `gemma4_moe.rs` — and its measured decode effect is +1.6…4.1%,
  which is not 28% either.)

**Left, in cost order:** the mlx-rs fork moved once in the window
(`4db2402d` → `f8cfdd88`, 2026-05-29, two commits both labelled CI fixes) and
has not moved since; and ~30 lumen commits touched shared decode code
(`native_cache.rs` 11, `native_quant.rs` 8, `native_attention.rs` 5). Attributing
further needs a rebuild at the old fork rev or a bisect, and each bisect step is
a full MLX rebuild.

**The reusable lesson is the metadata, not the culprit.** That row could not be
attributed because it records a number and nothing else: no prompt length, no
model build, no date. The bench's own default `MODEL_ID` is the scrubbed
placeholder `/path/to/models/gemma-4-26b-a4b-mlx-4bit`, so even the checkpoint
that produced 18.8 is unrecoverable — and six different Gemma 4 26B-A4B builds
exist on this machine now. Rows added below carry their conditions.

Getting there took four attempts, and the first three failed in instructive
ways. Blocked ON→OFF said OFF was 20–27% faster. The reversed order produced a
**93.5 s** run against a 1.5–3.5 s norm. Interleaved ABAB across processes, on a
quiet machine, gave 1 win ON / 2 OFF / 1 tie with a 2.7× spread *inside* each
side. Every one of those paid a fresh 26 B model load, cold page cache and
Metal pipeline compilation per sample — variance charged to the measurement and
far larger than the effect.

`examples/bench_env_flag_ab.rs` is the tool that worked: one model load, flag
flipped between `generate` calls, `decode_ms` only. Use it for any flag read via
`env::var` at its point of use — and **not** for a `OnceLock`-cached
`lumen_flags` flag, which latches on first read and would give a beautifully
tight "no difference" while measuring one side twice.

Two things it taught about measuring on this machine:

* **`min` is the estimator, not `mean` or `max`.** Contention only ever makes a
  sample slower, so the fastest run of each side is the least interfered with.
  The noise floor is `median/min` — the spread of the clean cluster.
* **A robust spread metric, because outliers happen.** 4 of 20 samples exceeded
  3× the median, one at 40×. An outlier-sensitive metric (`max/min`) reported
  4179% and made every verdict INCONCLUSIVE by construction; a tool that cannot
  conclude is not measuring anything.

  ⚠️ **Those stalls were self-inflicted, and this file said otherwise.** It
  originally read "stalls happen even when idle" — presented as a property of
  the machine. They were memory pressure: 36 GB of RAM, a 16 GB model resident,
  and a release-build `cargo xtask gate` started underneath it. The box
  eventually froze and rebooted. Read the same way, the gate slowing from 555s
  to 775s to 841s across the session was the same signal, and it was missed too.

  **So: never run a model bench and a build/gate at the same time.** One at a
  time, and wait. Run alone on a settled machine the gate finishes in ~810s and
  never takes the load above 6.4. If a measurement here shows 40× outliers,
  suspect your own concurrency before the hardware.

And read the `[runaway] … aborted` line first: unequal step counts make two runs
incomparable, which is the trap documented above and the one that broke this
experiment before thermals got a chance to.

`Qwen3.6-35B-A3B-mxfp4` needs the actual mxfp4 checkpoint —
`mlx-community/Qwen3.6-35B-A3B-mxfp4`, 18 GB, `mode: mxfp4 / bits 4 /
group_size 32`. An `affine` 4-bit build is not a substitute: different
bytes-per-token, so it would be compared against a number that only means
anything for mxfp4.

⚠️ **`bench_gemma4_native_e2e` does not run the `STEPS` you ask for.** On a
synthetic prompt the model degenerates and the runaway guard aborts — at step 32
of a requested 64, at both `PROMPT_LEN=8` and `512`. The per-step figure is still
sound (it is an average), but a request for 64 steps silently yields 32, so a
run is less thermally loaded than it looks. Check the `[runaway] … aborted` line
before comparing two runs of different lengths.

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

**Re-measured on a later tree**, same harness, same model, 20,000 tokens,
`--chunks 2048,1024 --gen 8`:

| prompt | chunk | prefill | memory after prefill |
|---|---|---|---|
| 20,000 tok | 2048 | 50.4 s | **1,959 MB** (was 2,556–2,621) |
| | 1024 | 49.7 s | **1,153 MB** (was 1,750–1,816) |

Memory is ~25–35% lower while the prefill time lands inside the original band,
so this is not a different workload being measured — something reduced the
per-chunk activation footprint between the two runs. The `+9 to +17%` time
penalty at 1024 did not reproduce here either (49.7 s vs 50.4 s, inside noise
for a two-point sample).

Both rows are kept. The point of a recorded number is to be re-measured, and a
release-checklist item that reads "does not regress" against a number nobody has
re-taken is checking against a memory rather than against the code. Output was
byte-identical across both chunk sizes, so the chunk-invariance argument still
holds.

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

### bf16 KV storage — DEFAULT ON (`LUMEN_MLX_KV_BF16=0` to revert)

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

#### Default flipped on

Memory, speed and quality all point the same way, so bf16 is now the default
storage dtype for the full-attention KV cache. `LUMEN_MLX_KV_BF16=0` restores
f32.

What changes for a user: generated text can differ from a pre-flip build. Every
divergence measured sat at a logit tie below the 1.5th percentile, so this is a
different-but-equivalent continuation rather than a worse one — but a 600-token
reply will differ from the old build's a fair fraction of the time, and anything
pinned to exact historical output (recorded transcripts, `LUMEN_MLX_GOLDEN_IN`
parity fixtures captured under f32) needs regenerating or running with the flag
off.

One hazard worth knowing: a KV cache **persisted** under one setting cannot be
extended under the other. `LUMEN_KV_DISK` is off by default so the exposure is
narrow, but `NativeKvCache::update_and_fetch` now rejects a dtype mismatch with
an actionable message rather than silently promoting — the failure would
otherwise surface far from its cause.

Scope: this is the Qwen3.5/3.6 full-attention path. Gemma 4 has its own cache
(`NativeKvCacheQuantized` and friends) and is untouched, as is the TurboQuant
full-attn variant.

The other lever 007 named — `LUMEN_QWEN35_PREFILL_CHUNK` — is covered above; it
is a knob, not a default change.

> **Measurement note on the tables above.** `get_active_memory()` after a
> single `clear_cache()` is not settled: `LUMEN_NATIVE_DEFER_CLEAR_CACHE` is on
> by default, so `remove_seq` hands the clear to a background worker (~45 ms)
> and a reading taken before it runs is high by a fixed ~127 MB. Both harnesses
> now loop with a 25 ms wait until two readings agree. The 007 per-run absolute
> figures were taken before that fix and carry the quantum as noise — which is
> why the fit reports a worst point 4.5-7.2% off. The fitted constants are
> unaffected and were confirmed independently: the settled harness reads
> 188.6 MB at N=1 / 2,048 slots against the fit's 191.4 MB prediction.
>
> Verified after the default flip, two passes each: unset → 121.4 / 121.5 MB
> (bf16), `LUMEN_MLX_KV_BF16=0` → 188.6 / 188.6 MB (f32), residual 0.0 MB.

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
