# Golden references

A golden nobody can regenerate rots the moment the model or the reference
implementation moves. That is not hypothetical here: the
`flux-scheduler-invariants` defect in `xtask/src/red_green.rs` was a test
comparing against `/tmp/klein_sigmas.bin`, a dev-session dump that had ceased to
exist, so it failed on every machine but the one it was written on.

So every file in this directory gets an entry below, and every entry answers
three questions: **what it pins**, **how it was produced**, and **how to
reproduce it**. An entry that cannot answer the third one says so in those
words, because a golden with an unknown provenance is a number nobody may
change and nobody can defend.

The generators live in `scripts/golden/`, committed for exactly this reason
while the rest of `scripts/` stays local (see `.gitignore`).

---

## `embedding_qwen3_0_6b_8bit.json` — 776 KB

**Pins:** the `/v1/embeddings` port. 25-item retrieval corpus, per-item
reference vectors, and reference P@1 / P@3 / MRR. Read by
`crates/lumen-mlx/examples/embedding_parity.rs`, which requires the MLX model to
reproduce all three on the same checkpoint:

1. per-item cosine ≥ 0.99 against the reference vector — the strict one, because
   two embedders can score identically on a 25-item eval while embedding into
   different spaces;
2. unit norm, because `/v1/embeddings` consumers treat dot as cosine;
3. retrieval metrics at least as good as the reference.

**Produced by:** the **candle** implementation of the embedding path, on
`mlx-community/Qwen3-Embedding-0.6B-8bit`.

**Reproducing it: not currently possible.** The candle backend was removed in
`7eacd3a` ("Remove the Candle backend; port /v1/embeddings to MLX"), and its
generator went with it. The file is a frozen artifact of a backend this repo no
longer contains.

That is a real limitation and worth stating plainly rather than leaving to be
discovered: this golden can be *checked against* but not *rebuilt*, so if it is
ever found to be wrong there is no way to produce a corrected version short of
reinstating candle or choosing a new reference. What it still buys is the thing
it was made for — proof that the MLX port did not silently change the embedding
space during the migration — and that value does not decay.

**If it ever needs replacing**, the honest options are, in order of preference:

1. Regenerate against `sentence-transformers` / HF `transformers` on the same
   checkpoint. This changes the reference implementation, so re-derive the
   metrics rather than assuming the old thresholds transfer.
2. Freeze the current MLX output as a *self*-golden. Cheaper, and strictly
   weaker: it pins against regression but no longer against the original
   candle-era behaviour, so it cannot catch a drift that already happened.

Option 2 is the trap this README exists to flag. Taking it silently would turn a
cross-implementation check into a self-consistency check while leaving the file
name and the test unchanged.

**Provenance note:** the file's `model_id_used` field previously recorded a
local HuggingFace cache path from the machine that generated it. It now records
the model id, which is the golden's actual identity — where it happened to be
cached is not part of what is being pinned. Found by `cargo xtask gate`'s
hygiene check.

---

## Adding a golden

1. Commit the **generator** to `scripts/golden/` in the same change. If the
   generator cannot be committed, say why in this file before merging.
2. Add an entry here with the three questions answered.
3. Keep it KB-scale. Token ids and checksums, not tensors — a golden that is too
   large to read is a golden nobody reviews.
4. Read it from this directory, never from `/tmp` or an env-supplied path that
   the repo does not produce.
