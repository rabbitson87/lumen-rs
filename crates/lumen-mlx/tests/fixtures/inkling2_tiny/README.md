# `inkling2_tiny` — a random-weights replica of an Inkling-2-Mini checkpoint (~4.9 MB)

**Pins:** the `inkling2` text model (`src/inkling2/{model,ops}.rs`) against the oMLX reference
implementation (oMLX 0.8.2, `omlx/patches/inkling2/`, Apache-2.0). `reference/outputs.safetensors`
holds the reference's logits for a 48-token prompt (a full prefill, and a 40-token prefill + 8
single-token decode steps), and the residual stream entering each of the 48 layer visits plus the
final-norm output. The tests in `src/inkling2/tests.rs` require the port to match to 5e-3 max
|logit diff| and cosine > 0.9999, the same top-1 at every position, and chunked prefill to agree
with a single prefill. Measured: a full prefill is **bit-identical** to the reference (max diff 0,
all 49 residual-stream tensors included); prefill + decode 1.5e-4 and chunked prefill 4.3e-4 (MLX
picks another matmul kernel for another row count, in the reference as well: its own prefill and
prefill + decode differ by 3.8e-4).

**What the replica exercises:** the shape of the real model with small widths. 24 layers, global
layers 5/11/17/23, the loop core 6..18 with 3 passes (48 layer visits, each with its own KV cache),
block size 6, hidden 64, 4 query and 2 key/value heads of 32, 16 experts with top-4 routing, NVFP4
experts (with per-expert tensor scales) in layers 0-19 and affine 4-bit group 64 in layers 20-23,
8-bit affine embedding, attention and `lm_head`, 250 live ids in a 256 vocabulary. The sliding
window (16), the global layers' relative band (24) and the log-scaling floor (8) are small against
the 48-token prompt so each binds, and the L2 clip thresholds are small so the clips are active.
The fixture script prints how far the reference's logits move when each mechanism is switched off
(5.1 to 5.9 for log scaling, both clips, the loop gain, the NVFP4 tensor scales and the sinks); a
mechanism that moved nothing would be invisible to the tests. Tensor names and the quantization
block are those of the real checkpoint. Scales and biases are float32 so the reference and the
port both run in float32.

**Two choices that make the gate sound** (both in `scripts/golden/inkling2_tiny_fixture.py`, with
the measurements):

- The loop controllers round their input and output through bf16 (`_mix_messages`), whatever the
  weight dtype. In a float32 replica that rounding is a staircase: two implementations that differ
  by 1e-7 occasionally land on different sides of a bf16 boundary, the element jumps by ~4e-3, and
  the 48 visits amplify it. The reference's own 48-row and 40-row prefills of the same 40 tokens
  differed by 7e-2 in the logits with the rounding and by 8e-5 without. So the end-to-end outputs
  are computed without the rounding on both sides (`LoadOptions::round_loop_mix = false` in the
  port), and the rounding is pinned separately, on identical inputs, by
  `reference/mix_messages.safetensors` (bit-identical, both modes, N = 3 and N = 1).
- On the GPU the reference mixes the depth sources in a Metal kernel; off the GPU it runs plain
  ops (the same function). The kernel's fast-math exp and reduction order differ from the plain
  ops by 8.5e-4 in the logits after 48 visits. The port is the plain-ops form, so the replica runs
  that form and the gate measures the port against the reference's own arithmetic.

**Produced by:** `scripts/golden/inkling2_tiny_fixture.py`, which imports the reference (copied
into a temporary package; the only stub is `omlx.custom_kernels.nax.is_nax_available`), builds the
model, quantizes it and writes this directory. Seeds are fixed.

**Reproduce:**

```bash
git -C <omlx> archive ade9f20 omlx/patches/inkling2 | tar -x -C <dir>
python scripts/golden/inkling2_tiny_fixture.py <dir> crates/lumen-mlx/tests/fixtures/inkling2_tiny
```

Needs `mlx` and `mlx-lm` (the committed files were produced with mlx 0.31.2 / mlx-lm 0.31.3 from
the repository's `.venv`). A rerun reproduces `config.json`, `model.safetensors` and
`reference/mix_messages.safetensors` byte for byte; `reference/outputs.safetensors` can move by
~1e-6 between MLX releases (kernel reduction order), which the tolerances above allow for.
