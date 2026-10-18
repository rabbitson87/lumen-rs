# `qwen4exp_tiny` — a random-weights replica of a Qwen3.8-Flash-Next checkpoint (~2.5 MB)

**Pins:** the `qwen4_exp` port (`src/qwen4_exp/`) against the reference
implementation. `reference/outputs.safetensors` holds the reference's logits for
a 48-token prompt (full prefill, and prefill + 8 single-token decode steps), the
residual stream after every layer, and the n-gram row ids. The tests in
`src/qwen4_exp/tests.rs` require the port to match to 5e-3 max |logit diff| and
cosine > 0.9999 (measured: 3e-6 and 1.000000), the same top-1 at every
position, and bit-identical results across chunked prefill, the file-backed
n-gram table and an expert slot pool.

**What the replica exercises:** 8 layers (Gated DeltaNet ×3 + attention ×1, twice),
16 experts per layer with top-4 routing, expert and shared-expert *biases*,
hyper-connections with 4 streams, the n-gram embedding with an EOS inside the
prompt, and the QSA indexer with a 16-key budget against a 48-token prompt (so
the sparse path decides the answer: switching it off moves the logits by 0.42).
Tensor names and the quantization layout (affine 4-bit group 32; router and
shared-expert gate 8-bit group 64) are those of the mlx-community build. Scales
and biases are float32 so the reference and the port both run in float32.

**Produced by:** `scripts/golden/qwen4exp_tiny_fixture.py`, which builds the model
with the vendored reference (Slotstream's `Tools/reference/qwen4_exp.py`,
github.com/carloslfu/slotstream, MIT), quantizes it and writes this directory.
Seeds are fixed.

**Reproduce:**

```bash
python scripts/golden/qwen4exp_tiny_fixture.py \
    <slotstream checkout>/Tools/reference crates/lumen-mlx/tests/fixtures/qwen4exp_tiny
```

Needs `mlx` and `mlx-lm` (any recent version; the oMLX venv has both). A rerun
should reproduce the same bytes; if MLX changes how it quantizes, the files change
and the tolerances above are what to re-check.
