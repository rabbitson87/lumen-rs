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

Needs `mlx` and `mlx-lm`. A rerun reproduces the same bytes **on the same MLX**: the
committed files were produced with mlx 0.32.4 / mlx-lm 0.31.4.dev from the oMLX
checkout's venv (`<omlx>/.venv/bin/python3`). Other releases (e.g. mlx 0.31.2) reproduce
`config.json`, `model.safetensors` and `mtp.*` exactly but move the reference outputs by
~1e-6 (kernel reduction order), so `reference/outputs.safetensors` and
`reference/mtp_outputs.safetensors` differ byte-wise. If MLX changes how it quantizes,
the weight files change and the tolerances above are what to re-check.

## MTP draft head

`mtp.safetensors`, `mtp.json` and `reference/mtp_outputs.safetensors` pin the MTP
(multi-token-prediction) head: one `full_attention` decoder layer (QSA attention, 16-expert
top-4 MoE, hyper-connections) plus the fusion layers (`fc_embedding`, `fc_hidden`,
`pre_fc_norm_*`) and the head's own hyper-connection mixer. They are produced by the last
section of the same script, with Slotstream's `Tools/reference/mtp_ref.py` (`MTPModule`),
random well-conditioned weights drawn from the same numpy stream (after every draw of the
main model, so the main-model files are unaffected) and the same quantization plan (affine
4-bit group 32; `mlp.gate` and `mlp.shared_expert_gate` 8-bit group 64; float32 scales and
biases). The embedding table and `lm_head` are the main model's.

- `mtp.safetensors` (78 tensors) names are the module's parameters prefixed `mtp.`, e.g.
  `mtp.fc_embedding.weight`, `mtp.layers.0.mlp.switch_mlp.gate_proj.weight`,
  `mtp.layers.0.self_attn.q_proj.scales`, `mtp.hyper_connection_mixer.hc_norm.weight`. The
  routed experts are already split into `switch_mlp.{gate,up,down}_proj` (the real checkpoint
  stores them fused; `scripts/qwen4_mtp_fetch.py` splits them, gate = first half). Norm weights
  are stored **as the module applies them** (`norms: "folded"`: the +1 of the HF
  zero-centred norms is already in the value).
- `mtp.json`: `{"norms": "folded", "group_size": 32, "bits": {<base>: bits}, "quantization":
  {"default": {group_size, bits}, "<base>": {group_size, bits}, ...}}`. `bits` lists every
  quantized base (name without `.weight`); `group_size` is the default and the
  `quantization` map gives per-base overrides (here the two 8-bit group-64 gates). The
  real-checkpoint sidecar written by `scripts/qwen4_mtp_fetch.py` has the same keys with
  group 64 everywhere and no `quantization` map.
- `reference/mtp_outputs.safetensors` (float32, `mtp_ids` int32). Entry `j` (1..47) fuses
  the backbone's pre-final-mixer multi stream at position `j-1` with the embedding of token
  `j`, so there are 47 entries and the head's own cache positions start at 0:
  `mtp_ids` [1,48] (same as `ids` in `outputs.safetensors`), `mtp_multi_in` [1,47,256]
  (`after_layer_7[:, :47]`), `mtp_embedded` [1,47,64] (main model's dequantized embedding rows
  of `ids[:, 1:]`), prefill with no cache: `mtp_sample_prefill` [1,47,64],
  `mtp_multi_prefill` [1,47,256], `mtp_logits_prefill` [1,47,128] (`lm_head(sample)`);
  and the same entries through a fresh attention cache (with its indexer cache) as one
  39-entry chunk plus 8 single-entry steps: `mtp_logits_chunk` [1,39,128],
  `mtp_logits_steps` [1,8,128], `mtp_multi_steps` [1,8,256]. The mask is `causal` for
  multi-entry calls and none for a single entry (the QSA indexer's boolean mask replaces it
  once more than 16 keys are visible). Incremental vs prefill agree to 5.5e-7 (logits) and
  1.6e-6 (multi); the QSA sparse path matters (dense attention moves the logits by 8.8e-2)
  and a 1e-6 relative nudge of the inputs moves them by 7e-7, so no output hinges on a
  near-tie.
