#!/usr/bin/env python3
"""Tiny random-weights replica of an Inkling-2-Mini (`inkling2_loop`) MLX checkpoint.

Builds the model with the oMLX reference implementation (oMLX 0.8.2,
`omlx/patches/inkling2/`, Apache-2.0), quantizes it the way the mlx-community
oQ4e build is quantized (embedding, attention and `lm_head` 8-bit affine group 64;
routed experts NVFP4 in layers 0-19 and affine 4-bit group 64 in layers 20-23), and writes:

  <out>/config.json                     same schema as the real checkpoint's text_config and
                                        quantization block, tiny sizes (no vision, no MTP)
  <out>/model.safetensors               the runtime-layout tensor names of the real checkpoint
  <out>/reference/outputs.safetensors   ids, reference logits (full prefill; prefill + 8 decode
                                        steps), and the residual stream entering every layer
                                        visit plus the final-norm output, all float32

The shape mirrors the real model: 24 layers, global layers 5/11/17/23, loop core 6..18 with
3 passes (48 layer visits), block size 6. Only the widths are small. The window (16), the
relative band of the global layers (24) and the log-scaling floor (8) are small against the
48-token prompt so each of them binds, and the clip thresholds are small so the L2 clips are
active. The script prints how far the reference logits move when each mechanism is switched
off; a mechanism that moves nothing would be hidden from the parity test.

Scales and biases are float32, so the reference and the Rust port both run in float32 and agree
to rounding error. The reference's explicit bf16 round trips in `_mix_messages` are switched
off for the end-to-end outputs (see the comment at the top of the code: in float32 they make the
network discontinuous) and pinned on their own in `reference/mix_messages.safetensors`.

usage: python scripts/golden/inkling2_tiny_fixture.py <dir containing omlx/patches/inkling2> \\
           crates/lumen-mlx/tests/fixtures/inkling2_tiny
       Needs mlx and mlx-lm. oMLX's working tree may predate 0.8.2; extract the files from git:
           git -C <omlx> archive ade9f20 omlx/patches/inkling2 | tar -x -C <dir>

The output is committed: the Rust tests (src/inkling2/tests.rs) read it, so this only needs
re-running if the reference or the replica's shape changes. Seeds are fixed. The reference
outputs move by ~1e-6 between mlx releases (kernel reduction order); the committed ones were
produced with mlx 0.31.2 / mlx-lm 0.31.3.
"""
import importlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

ref_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

SEED = 11
rng = np.random.default_rng(SEED)
mx.random.seed(SEED)


def load_reference(root: Path):
    """Import oMLX's `inkling2/model.py` without importing the rest of oMLX.

    `kernels.py` reaches for `omlx.custom_kernels.nax.is_nax_available`; a stub package with
    that one function stands in for it. The reference files are copied, not edited."""
    src = root / "omlx" / "patches" / "inkling2"
    if not (src / "model.py").is_file():
        sys.exit(f"{src}/model.py is missing")
    tmp = Path(tempfile.mkdtemp(prefix="inkling2_ref_"))
    pkg = tmp / "omlx_ref"
    for d in (pkg, pkg / "patches", pkg / "custom_kernels"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")
    (pkg / "custom_kernels" / "nax.py").write_text("def is_nax_available():\n    return False\n")
    shutil.copytree(src, pkg / "patches" / "inkling2")
    sys.path.insert(0, str(tmp))
    return importlib.import_module("omlx_ref.patches.inkling2.model")


ref = load_reference(ref_root)

# The loop controllers round their inputs to bf16 and back (`_mix_messages`), whatever the weight
# dtype. In a float32 replica that rounding is a staircase: two implementations that differ by
# 1e-7 occasionally land on different sides of a bf16 boundary, the element jumps by ~4e-3, and
# 48 layer visits amplify it (measured: the reference's own 48-row and 40-row prefills of the same
# 40 tokens differ by 7e-2 in the logits; without the rounding, 8e-5). So the end-to-end replica
# runs without the rounding, on both sides, and the rounding is pinned separately on identical
# inputs (`reference/mix_messages.safetensors`, written below).
_mix_with_bf16_rounding = ref._mix_messages


def _mix_without_rounding(x, message, down, gain_forget):
    n, b, length, hidden = x.shape
    rows = b * length
    z = mx.stack([ref._unit_rms(x), ref._unit_rms(message)], axis=1).reshape(2 * n, rows, hidden)
    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)
    gates = (low @ gain_forget).reshape(n, b, length, 2 * hidden)
    gain = mx.tanh(gates[..., :hidden])
    forget = mx.tanh(gates[..., hidden:])
    return (1.0 - forget) * x + gain * message


ref._mix_messages = _mix_without_rounding


# On the GPU the reference mixes the depth sources in a Metal kernel (`kernels.attn_res_mix`); off
# the GPU it runs the plain ops below, which are the same function (the first branch of that
# function, copied). The kernel's fast-math exp and reduction order differ from the plain ops by
# 8.5e-4 in this replica's logits after 48 visits (measured), noise a parity gate would have to
# absorb. The Rust port is the plain-ops form, so the replica runs that form: the gate then
# measures the port against the reference's own arithmetic.
def _attn_res_mix_plain_ops(blocks, partial_n, query, scale):
    src = mx.stack([*blocks, partial_n], axis=-2)
    logits = (src @ query.T) * scale
    w = mx.softmax(logits, axis=-2, precise=True)
    return w.swapaxes(-1, -2) @ src


ref.kernels.attn_res_mix = _attn_res_mix_plain_ops

HIDDEN = 64
LAYERS = 24
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
D_REL = 4
EXPERTS, TOP_K, INTER = 16, 4, 64
VOCAB, UNPADDED = 256, 250
GATE_RANK = 8
CORE_START, CORE_END, PASSES = 6, 18, 3
GLOBAL_LAYERS = [5, 11, 17, 23]
LOCAL_LAYERS = [i for i in range(LAYERS) if i not in GLOBAL_LAYERS]

TEXT = dict(
    model_type="inkling2_loop_text",
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    vocab_size=VOCAB,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    swa_num_attention_heads=HEADS,
    swa_num_key_value_heads=KV_HEADS,
    swa_head_dim=HEAD_DIM,
    d_rel=D_REL,
    rel_extent=24,
    sliding_window_size=16,
    log_scaling_n_floor=8,
    log_scaling_alpha=0.1,
    rms_norm_eps=1e-6,
    qk_l2_clip_threshold=3.0,
    rel_l2_clip_threshold=1.5,
    local_layer_ids=LOCAL_LAYERS,
    sconv_kernel_size=4,
    unpadded_vocab_size=UNPADDED,
    final_logit_softcapping=30.0,
    n_layers_per_attn_res_block=6,
    n_routed_experts=EXPERTS,
    num_experts_per_tok=TOP_K,
    n_shared_experts=0,
    intermediate_size=INTER,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    route_scale=8.0,
    use_gate_bias=True,
    loop_config=dict(
        core_start=CORE_START,
        core_end=CORE_END,
        num_passes=PASSES,
        gate_rank=GATE_RANK,
        is_varloop=False,
    ),
)

# The real checkpoint's block: affine 4-bit group 64 as the default, 8-bit group 64 for the
# embedding, `lm_head` and every attention projection, NVFP4 for the experts of layers 0-19.
AFFINE8 = {"bits": 8, "group_size": 64, "mode": "affine"}
NVFP4 = {"bits": 4, "group_size": 16, "mode": "nvfp4"}
QUANT = {"group_size": 64, "bits": 4, "mode": "affine"}
QUANT["model.embed"] = dict(AFFINE8)
QUANT["lm_head"] = dict(AFFINE8)
for i in range(LAYERS):
    if i in GLOBAL_LAYERS:
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            QUANT[f"model.layers.{i}.attn.{n}"] = dict(AFFINE8)
    else:
        for n in ("in_proj", "o_proj"):
            QUANT[f"model.layers.{i}.attn.{n}"] = dict(AFFINE8)
    if i < 20:
        for w in ("w13", "w2"):
            QUANT[f"model.layers.{i}.mlp.experts.{w}"] = dict(NVFP4)

args = ref.ModelArgs(model_type="inkling2_loop", text_config=TEXT, quantization=QUANT)
model = ref.Model(args)


def normal(shape, std):
    return mx.array(rng.normal(0.0, std, size=shape).astype(np.float32))


def uniform(shape, lo, hi):
    return mx.array(rng.uniform(lo, hi, size=shape).astype(np.float32))


def init_float_params(module, prefix=""):
    """Replace every float parameter by a random, well-conditioned f32 value. The default
    inits leave norms at one and the conv taps at zero, which would hide wiring mistakes behind
    symmetric values."""
    flat = dict(tree_flatten(module.parameters()))
    new = {}
    for name, p in flat.items():
        shape = p.shape
        leaf = name.rsplit(".", 1)[-1]
        if "mlp.experts" in name:
            continue  # filled below with quantized carriers
        if name.endswith(".query"):
            v = normal(shape, 1.5)
        elif leaf == "router_weight":
            v = normal(shape, 0.6)
        elif leaf == "router_bias":
            v = normal(shape, 0.1)
        elif leaf == "loop_down":
            v = normal(shape, 1.0 / math.sqrt(HIDDEN))
        elif leaf == "loop_gain_forget":
            v = normal(shape, 0.7 / math.sqrt(GATE_RANK))
        elif leaf in ("attn_conv", "mlp_conv", "k_conv", "v_conv", "k_static", "v_static"):
            v = normal(shape, 0.4)
        elif leaf in ("q_scale", "r_scale"):
            v = uniform(shape, 0.8, 1.2)
        elif leaf == "rel_proj":
            v = normal(shape, 0.6)
        elif leaf == "weight" and len(shape) == 1:
            v = uniform(shape, 0.8, 1.2)  # RMSNorm weight
        elif leaf == "weight" and len(shape) == 2:
            fan_in = shape[1] if "embed" not in name or "norm" in name else 1
            v = normal(shape, 1.0 / math.sqrt(fan_in))
        else:
            sys.exit(f"unhandled parameter {name} {shape}")
        new[name] = v.astype(mx.float32)
    module.load_weights(list(new.items()), strict=False)


init_float_params(model)

# 8-bit affine group 64 for the embedding, `lm_head` and the attention projections. The experts
# own their quantization (`ExpertProj.to_quantized` returns itself).
nn.quantize(
    model,
    group_size=64,
    bits=8,
    class_predicate=lambda path, m: isinstance(m, (nn.Linear, nn.Embedding)),
)


def quantize_stack(w, spec):
    out = mx.quantize(w, group_size=spec["group_size"], bits=spec["bits"], mode=spec["mode"])
    return out[0], out[1], (out[2] if len(out) > 2 else None)


for i, layer in enumerate(model.model.layers):
    ex = layer.mlp.experts
    for which, proj, (rows, cols) in (("w13", ex.w13, (2 * INTER, HIDDEN)), ("w2", ex.w2, (HIDDEN, INTER))):
        spec = QUANT.get(f"model.layers.{i}.mlp.experts.{which}") or {
            k: QUANT[k] for k in ("group_size", "bits", "mode")
        }
        assert proj.mode == spec["mode"], (i, which, proj.mode, spec)
        w = normal((EXPERTS, rows, cols), 1.2 / math.sqrt(cols))
        wq, scales, biases = quantize_stack(w, spec)
        proj.weight = wq
        proj.scales = scales.astype(mx.float32) if spec["mode"] == "affine" else scales
        if spec["mode"] == "affine":
            proj.biases = biases.astype(mx.float32)
        if spec["mode"] == "nvfp4":
            # The per-expert tensor scale: `global_scale / (448 * 6)` is applied after the
            # matmul, so draw it around 1 to make the factor matter.
            proj.global_scale = (uniform((EXPERTS,), 0.8, 1.25) * (448.0 * 6.0)).astype(mx.float32)

mx.eval(model.parameters())
model.eval()

# ---- reference runs ---------------------------------------------------------------------------

N_PROMPT, N_DECODE = 40, 8
ids_np = rng.integers(0, UNPADDED, size=N_PROMPT + N_DECODE).astype(np.int32)
ids = mx.array(ids_np)[None]
visits = model.model.num_visits
assert visits == 48, visits

capture = {i: None for i in range(visits + 1)}
cache = model.make_cache()
hidden = model.model(ids, cache, capture=capture)
logits_full = model.logits_from_hidden(hidden)
mx.eval(logits_full, *[v for v in capture.values()])

plain = model(ids, cache=model.make_cache())
mx.eval(plain)
assert float(mx.max(mx.abs(mx.where(mx.isinf(plain), 0, plain - logits_full)))) == 0.0

cache = model.make_cache()
parts = [model(ids[:, :N_PROMPT], cache=cache)]
for t in range(N_PROMPT, N_PROMPT + N_DECODE):
    parts.append(model(ids[:, t : t + 1], cache=cache))
logits_split = mx.concatenate(parts, axis=1)
mx.eval(logits_split)


def finite_part(x):
    return x[..., :UNPADDED]


def spread(a, b):
    return float(mx.max(mx.abs(finite_part(a) - finite_part(b))))


print(f"prefill vs prefill+decode: max |diff| {spread(logits_full, logits_split):.3e}")
assert bool(mx.all(logits_full[..., UNPADDED:] == -mx.inf))
assert bool(mx.all(mx.isfinite(finite_part(logits_full))))
print("top-1 distinct ids:", len(set(np.argmax(np.array(logits_full[0, :, :UNPADDED]), -1).tolist())))

# ---- does every mechanism bind? ---------------------------------------------------------------


def logits_with(edit):
    undo = edit()
    out = model(ids, cache=model.make_cache())
    mx.eval(out)
    undo()
    return out


def attn_layers():
    return [l.attn for l in model.model.layers]


def no_tau():
    olds = [a.log_alpha for a in attn_layers()]
    for a in attn_layers():
        a.log_alpha = 0.0

    def undo():
        for a, o in zip(attn_layers(), olds):
            a.log_alpha = o

    return undo


def no_qk_clip():
    old = [(a._qk_clip, a._r_clip) for a in attn_layers()]
    for a in attn_layers():
        a._qk_clip = mx.array(1e9, dtype=mx.float32)

    def undo():
        for a, (q, r) in zip(attn_layers(), old):
            a._qk_clip, a._r_clip = q, r

    return undo


def no_rel_clip():
    old = [a._r_clip for a in attn_layers()]
    for a in attn_layers():
        a._r_clip = mx.array(1e9, dtype=mx.float32)

    def undo():
        for a, o in zip(attn_layers(), old):
            a._r_clip = o

    return undo


def no_loop_gain():
    layers = [model.model.layers[i] for i in range(CORE_START, CORE_END)]
    old = [l.loop_gain_forget for l in layers]
    for l in layers:
        l.loop_gain_forget = mx.zeros_like(l.loop_gain_forget)

    def undo():
        for l, o in zip(layers, old):
            l.loop_gain_forget = o

    return undo


def no_global_scale():
    olds = []
    for l in model.model.layers[:20]:
        olds.append((l.mlp.experts.w13.global_scale, l.mlp.experts.w2.global_scale))
        l.mlp.experts.w13.global_scale = mx.full((EXPERTS,), 448.0 * 6.0, mx.float32)
        l.mlp.experts.w2.global_scale = mx.full((EXPERTS,), 448.0 * 6.0, mx.float32)

    def undo():
        for l, (a, b) in zip(model.model.layers[:20], olds):
            l.mlp.experts.w13.global_scale, l.mlp.experts.w2.global_scale = a, b

    return undo


def no_sinks():
    # A sink logit of -inf removes the sink key: the softmax then runs over the keys alone.
    # Done through the projection output would change other things, so patch `attend`.
    orig = ref.kernels.banded_sink_attention

    def patched(q, k, v, rel, sink, **kw):
        return orig(q, k, v, rel, mx.full(sink.shape, -1e30, sink.dtype), **kw)

    ref.kernels.banded_sink_attention = patched
    return lambda: setattr(ref.kernels, "banded_sink_attention", orig)


for label, edit in (
    ("log scaling (tau) off", no_tau),
    ("q/k L2 clip off", no_qk_clip),
    ("relative L2 clip off", no_rel_clip),
    ("loop gain zeroed", no_loop_gain),
    ("NVFP4 tensor scales = 1", no_global_scale),
    ("sink logits removed", no_sinks),
):
    moved = spread(logits_full, logits_with(edit))
    print(f"  {label:28s} moves the logits by {moved:.3e}")
    assert moved > 1e-3, f"{label}: the replica does not exercise this mechanism"

# ---- write ------------------------------------------------------------------------------------

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "reference").mkdir(exist_ok=True)

config = {
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 7,
    "text_config": TEXT,
    "quantization": QUANT,
    "quantization_config": QUANT,
}
(out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

params = dict(tree_flatten(model.parameters()))
mx.save_safetensors(str(out_dir / "model.safetensors"), params)

# The loop controllers on fixed inputs, with the reference's bf16 rounding and without. The Rust
# test feeds the same arrays to its own implementation, so no rounding boundary can be crossed
# by an upstream difference. N = 3 is the q/k/v call, N = 1 the MLP call.
mix = {}
for n in (3, 1):
    x = normal((n, 1, 5, HIDDEN), 1.0)
    msg = normal((n, 1, 5, HIDDEN), 1.0)
    down = normal((n, 2, HIDDEN, GATE_RANK), 1.0 / math.sqrt(HIDDEN))
    gf = normal((n, GATE_RANK, 2 * HIDDEN), 0.7 / math.sqrt(GATE_RANK))
    mix[f"x{n}"], mix[f"m{n}"], mix[f"down{n}"], mix[f"gf{n}"] = x, msg, down, gf
    mix[f"out{n}_bf16"] = _mix_with_bf16_rounding(x, msg, down, gf)
    mix[f"out{n}_plain"] = _mix_without_rounding(x, msg, down, gf)
    spread_bf16 = float(mx.max(mx.abs(mix[f"out{n}_bf16"] - mix[f"out{n}_plain"])))
    print(f"  mix_messages N={n}: bf16 rounding moves the output by {spread_bf16:.3e}")
    assert spread_bf16 > 1e-4, "the rounding must be visible in the unit fixture"
mx.save_safetensors(str(out_dir / "reference" / "mix_messages.safetensors"), mix)

outputs = {
    "ids": mx.array(ids_np),
    "logits_full": logits_full[0],
    "logits_split": logits_split[0],
}
for i in range(visits + 1):
    outputs[f"visit_{i:02d}"] = capture[i][0].astype(mx.float32)
mx.save_safetensors(str(out_dir / "reference" / "outputs.safetensors"), outputs)
print(f"wrote {out_dir} ({len(params)} tensors, {visits} visits)")
