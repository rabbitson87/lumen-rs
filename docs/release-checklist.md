# Release checklist

SQLite runs a ~200-item checklist before a release. This is the same idea at
this project's scale: the things that are **too slow for a pre-push gate** but
too important to skip before shipping.

`cargo xtask gate` covers the fast half and runs on every push. Everything here
is what it deliberately leaves out — soak-scale work, GPU-hardware checks, and
the judgement calls a command cannot make.

Run in order; the cheap and the automatic first.

## 1. Gates

- [ ] `cargo xtask gate` green from a **clean tree** (`git status` empty).
      A dirty tree can hide a file that was never committed.
- [ ] `cargo xtask red-green` — every defect PASS. **Zero VACUOUS, zero
      BROKEN.** A VACUOUS verdict means a guard passes with its fix reverted,
      which is a guard that would not have caught its own defect.
- [ ] `cargo xtask flags --check` clean, and `docs/env-flags.md` regenerated if
      any flag was added or its doc comment changed.

## 2. Equivalence and soak

- [ ] `cargo xtask flags` — the full suite with **every `Optimization` flag
      flipped**, identical results. An optimization that changes output is
      either misclassified or a bug; this is the only thing that tells them
      apart.
- [ ] `cargo xtask fuzz --all --minutes 30` — no crashes. Any crash found:
      commit the input to `fuzz/seeds/`, fix, and add a `DEFECTS` entry (see
      the regression policy below) **before** the release.
- [ ] `cargo xtask coverage` — one-sided branch count has not **risen**. It is
      allowed to stay flat; it is not allowed to grow silently, because a new
      untested branch is exactly what this measures.

## 3. Hardware and memory

- [ ] `cargo xtask test --validate` — Metal API + shader validation clean.
      This is the only check that catches an out-of-bounds buffer binding, and
      there is no substitute: an out-of-bounds read on a shared heap returns
      plausible numbers rather than crashing. Slow by roughly an order of
      magnitude; that is why it is here and not in the gate.
- [ ] Miri over `lumen-core`, the FFI-free crate and so the only one Miri can
      reason about. The command is not the obvious one, and both differences
      were found by running it rather than by reading about it:

      ```
      MIRIFLAGS="-Zmiri-disable-isolation" \
        cargo +nightly miri test -p lumen-core --lib -- --skip mtp_procrustes
      ```

      * `-Zmiri-disable-isolation` — three round-trip tests write a scratch file,
        and Miri blocks filesystem access by default. Without the flag the run
        stops at the first `File::create` and reports a failure that has nothing
        to do with memory.
      * `--skip mtp_procrustes` — those tests call `faer`'s SVD, which uses the
        AArch64 NEON intrinsic `llvm.aarch64.neon.fmaxnmv.f64.v2f64`. Miri does
        not emulate it and reports **`unsupported operation`**, which is a Miri
        limitation and not a finding. Skipping is the honest response; treating
        the abort as a failure would be as wrong as treating it as a pass.

      So Miri covers `lumen-core` **minus the faer-backed linear algebra**. That
      exclusion is the same kind as the `.metal` shaders in Phase 4.1: named,
      with the reason, rather than quietly absent.
- [ ] Performance within the bands in `docs/maintainer-workflow.md` §8, on an
      **idle** machine. A loaded machine has produced false regressions three
      times (`bf16-out-dispatch`, `dense-shapes-on-qmv-fast`, and the
      `mxfp4_bf16_in_parity` ratio) — measure twice before believing one.
- [ ] Long-prompt memory: a 20K-token prefill does not regress peak RSS.
      `LUMEN_QWEN35_PREFILL_CHUNK` is a memory↔latency exchange rate, not a
      free knob — see §9 of the maintainer workflow before touching it.

## 4. Hygiene and packaging

- [ ] `git grep "/Users/sonheesung"` empty outside the workflow doc that defines
      the rule. (The gate checks this; re-run it here on the clean tree.)
- [ ] `git grep "hsng95@gmail.com"` empty outside `LICENSE`.
- [ ] Fork SHAs pinned and reachable: `mlx-rs` / `mlx-sys` `rev =` in
      `crates/lumen-mlx/Cargo.toml` resolve from a clean clone, and the
      workspace `[patch]` overrides are **commented out**.
- [ ] `cargo build -p lumen-server --features mlx-native-metal --release` from a
      clean `target/` for the crate — a warm cache can hide a missing
      `required-features`.

## 5. Behaviour, by hand

Automation cannot tell you the model still answers well. Two requests against a
local model, one of each shape:

- [ ] A plain chat request — coherent, terminates on its own, no repetition.
- [ ] A tool-calling request — the call is well-formed, names a declared tool,
      and its arguments parse.
- [ ] A second turn on the same conversation — the prefix cache hits (check the
      log) and the answer still makes sense. A prefix-cache bug looks like a
      *quality* regression, not an error.

---

## Regression policy

**No bugfix lands without a `DEFECTS` entry in `xtask/src/red_green.rs` that
proves red→green.**

This is SQLite's most-cited practice and the machinery for it already exists
here; only the rule was missing. The entry needs three things:

1. **The symptom the defect produced in production** — not what the code did
   wrong, what the user saw. `--list` doubles as the evidence that each guard
   was earned rather than imagined.
2. **A mutation that reverts the fix in place.** Both sides must be non-empty;
   express a deletion as a sentinel comment, because the reverse direction
   searches for the replacement and an empty needle matches everywhere.
3. **A guard test that goes RED under that mutation.** If it stays green the
   harness reports VACUOUS and the run fails — which has already caught a guard
   whose invariant excluded the very input that triggered the defect.

Why it is worth the friction: a regression test written *after* a fix is worth
nothing until you show it fails without the fix, and that is the easy step to
skip precisely when the fix already works.

Two things that make the rule cheap to follow:

- Anchors are indentation-sensitive. Moving code between modules changes its
  nesting, so `find`/`replace` strings need re-indenting — `occurrences` then
  fails loudly rather than silently matching nothing.
- A guard on the ungated surface (`mlx_ungated_test`) builds in seconds where an
  `mlx-native` one takes minutes. Prefer it when the defect is in pure logic.
