#!/usr/bin/env python3
"""Tiny random-weights replica of Inkling 2's DSpark draft head, run by the oMLX reference.

usage: python scripts/golden/inkling2_dspark_fixture.py <dir containing omlx/> \\
           crates/lumen-mlx/tests/fixtures/inkling2_tiny crates/lumen-mlx/tests/fixtures/inkling2_dspark

The reference is oMLX 0.8.2 (`ade9f20`, Apache-2.0): `omlx/patches/inkling2/dspark.py` (the draft
model and its Markov head) and `omlx/speculative/dflash/drafter.py` (`DFlashDrafter._draft_batched`,
the draft forward), imported from a copy of those files with stubs for the parts of oMLX they
reach for and a single draft row never uses (the batch generator's samplers, the fused verify
kernels, the Metal selection kernels, mlx-vlm's RoPE helper). The reference files are not edited.
Extract them from git if the working tree predates 0.8.2:
    git -C <omlx> archive ade9f20 omlx/patches/inkling2 omlx/speculative | tar -x -C <dir>

The target the draft binds to is the tiny text replica (`inkling2_tiny`): its embedding, final
norm and LM head are the quantized ones the Rust tests load, so the draft's logits and the Markov
head's fold see the same numbers on both sides. The draft's own weights are random and float32 and
the ring is kept in float32 (`context_dtype`), so the port and the reference agree to rounding.

Writes into the output directory:

  config.json                 `mtp_config` of the replica (same schema as the real checkpoint's)
  mtp.safetensors             the draft's tensors under the checkpoint's names, float32
  reference/cycles.safetensors
      `n_cycles`, and per cycle `c`: `c.taps` (the new context rows, `[1, n, 4 * hidden]`),
      `c.anchor` (the newest committed token), `c.total` (the anchor's absolute position),
      `c.ring_k.<layer>` and `c.ring_v.<layer>` (the context K/V the next block attends to,
      `[1, kv_heads, rows, head_dim]`, oldest row first, RoPE applied), `c.positions`,
      `c.hidden` (the block's normalized output past the anchor), `c.base_logits`,
      `c.step_logits` (after the Markov head, softcap and vocabulary mask), `c.tokens` (the drafts)

Five cycles: the first seeds more rows than the window keeps, the later ones add 3, 1, 8 and 5.
The script prints how far the drafts move when each mechanism is switched off.
"""
import importlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

ref_root, tiny, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rng = np.random.default_rng(31)
mx.random.seed(31)


def load_reference(root: Path):
    """The draft model and drafter modules, without the rest of oMLX."""
    tmp = Path(tempfile.mkdtemp(prefix="inkling2_dspark_ref_"))
    pkg = tmp / "omlx"
    shutil.copytree(root / "omlx" / "patches" / "inkling2", pkg / "patches" / "inkling2")
    for name in ("variants.py", "selection.py", "drafter.py"):
        (pkg / "speculative" / "dflash").mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "omlx" / "speculative" / "dflash" / name, pkg / "speculative" / "dflash" / name)
    for d in (pkg, pkg / "patches", pkg / "speculative", pkg / "speculative" / "dflash"):
        (d / "__init__.py").write_text("")
    # `inkling2/__init__.py` is replaced by the one name the draft needs from it.
    (pkg / "patches" / "inkling2" / "__init__.py").write_text(
        "INKLING2_MODEL_TYPES = ('inkling2_loop', 'inkling2')\n"
    )
    stubs = {
        "omlx.patches.mlx_lm_mtp": "",
        "omlx.patches.mlx_lm_mtp.batch_generator": (
            "def _is_greedy(gen_batch):\n    return True\n"
            "def _resolve_draft_sampler(gen_batch, state):\n    return None\n"
            "class SparseDraftQ:\n    pass\n"
        ),
        "omlx.patches.qwen35_verify_qmm": (
            "def set_verify_qmm_armed(flag):\n    pass\n"
            "def sg8_swiglu_eligible(gate, up, x):\n    return False\n"
        ),
        "omlx.utils": "",
        "omlx.utils.sampling": "def top_k_indices(x, k):\n    raise NotImplementedError\n",
        "omlx.speculative.dflash.kernels": (
            "def conv_prepare(*a, **k):\n    raise NotImplementedError\n"
            "def conv_finish(*a, **k):\n    raise NotImplementedError\n"
            "def select_kernel():\n    raise NotImplementedError\n"
        ),
    }
    for name, body in stubs.items():
        path = pkg.joinpath(*name.split(".")[1:])
        if body or name.endswith(("mlx_lm_mtp", "utils")):
            if name in ("omlx.patches.mlx_lm_mtp", "omlx.utils"):
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").write_text(body)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".py").write_text(body)
    # mlx-vlm's RoPE helper is only reached by `apply_partial_rope`, which a draft forward never calls.
    vlm = types.ModuleType("mlx_vlm")
    models = types.ModuleType("mlx_vlm.models")
    rope_utils = types.ModuleType("mlx_vlm.models.rope_utils")
    rope_utils.initialize_rope = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError)
    sys.modules.update({"mlx_vlm": vlm, "mlx_vlm.models": models, "mlx_vlm.models.rope_utils": rope_utils})
    sys.path.insert(0, str(tmp))
    dspark = importlib.import_module("omlx.patches.inkling2.dspark")
    drafter = importlib.import_module("omlx.speculative.dflash.drafter")
    return dspark, drafter


dspark, drafter_mod = load_reference(ref_root)

# ---- the target the draft binds to: the tiny text replica's embedding, norm and head ----------
tiny_cfg = json.loads((tiny / "config.json").read_text())
text = tiny_cfg["text_config"]
H, V = int(text["hidden_size"]), int(text["vocab_size"])
UNPADDED = int(text.get("unpadded_vocab_size") or V)
SOFTCAP = float(text.get("final_logit_softcapping") or 0)
tensors = mx.load(str(tiny / "model.safetensors"))
quant = tiny_cfg["quantization"]["model.embed"]
GROUP, BITS = int(quant["group_size"]), int(quant["bits"])

embed = nn.QuantizedEmbedding(V, H, group_size=GROUP, bits=BITS)
embed.update(
    {"weight": tensors["model.embed.weight"], "scales": tensors["model.embed.scales"], "biases": tensors["model.embed.biases"]}
)
embed_norm = nn.RMSNorm(H, eps=float(text["rms_norm_eps"]))
embed_norm.weight = tensors["model.embed_norm.weight"]
lm_head = nn.QuantizedLinear(H, V, bias=False, group_size=GROUP, bits=BITS)
lm_head.update(
    {"weight": tensors["lm_head.weight"], "scales": tensors["lm_head.scales"], "biases": tensors["lm_head.biases"]}
)
target = types.SimpleNamespace(model=types.SimpleNamespace(embed=embed, embed_norm=embed_norm), lm_head=lm_head)

# ---- the replica ------------------------------------------------------------------------------
MTP = {
    "hidden_size": H,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 48,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 12,
    "block_size": 8,
    "markov_rank": 8,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 23, 41, 48],
}
full_cfg = {"model_type": "inkling2_loop", "text_config": text, "mtp_config": MTP}
config = dspark.Inkling2DSparkConfig.from_target_config(full_cfg)
model = dspark.Inkling2DSparkModel(config)
model.context_dtype = mx.float32  # the ring in float32, so the port and the reference can agree

params = {}
for name, value in tree_flatten(model.parameters()):
    shape = value.shape
    if name == "fuse_weight":
        data = rng.standard_normal(shape)
    elif name.endswith("norm.weight") or name.endswith("dflare_norm.weight"):
        data = 1.0 + 0.1 * rng.standard_normal(shape)
    elif name.endswith(".bias"):
        data = 0.1 * rng.standard_normal(shape)
    elif name == "mask_embed":
        data = 0.5 * rng.standard_normal(shape)
    elif name.endswith("markov_w1.weight"):
        data = 0.5 * rng.standard_normal(shape)
    else:
        data = rng.standard_normal(shape) / np.sqrt(shape[-1])
    params[name] = mx.array(data.astype(np.float32))
model.update(tree_unflatten(list(params.items())))
mx.eval(model.parameters())
model.bind(target)

out.mkdir(parents=True, exist_ok=True)
(out / "reference").mkdir(exist_ok=True)
(out / "config.json").write_text(json.dumps({"mtp_config": MTP}, indent=1) + "\n")
mx.save_safetensors(str(out / "mtp.safetensors"), params)

# ---- record the intermediates of every draft forward ----------------------------------------
recorded = {}
_orig_logits = model._logits


def _record_logits(hidden):
    logits = _orig_logits(hidden)
    recorded["hidden"], recorded["base_logits"] = hidden, logits
    return logits


model._logits = _record_logits
_orig_step = model.markov_head.apply_step_logits
steps = []


def _record_step(base_logits, prev_tokens):
    result = _orig_step(base_logits, prev_tokens)
    steps.append(result)
    return result


model.markov_head.apply_step_logits = _record_step


def run_cycles(model, captured_rows, anchors, depth=None):
    """Seed the first cycle's rows, then draft each cycle. Returns one record per cycle."""
    drafter = drafter_mod.DFlashDrafter(
        model, block_size=config.block_size, source_path="", adaptive=False, mask_embedding=model.mask_embed
    )
    state = types.SimpleNamespace(uid=0, depth=depth or config.block_size - 1, controller=None, stats=None)
    records, total = [], 0
    for c, (rows, anchor) in enumerate(zip(captured_rows, anchors)):
        steps.clear()
        taps = [rows[..., i * H : (i + 1) * H] for i in range(4)]
        total += rows.shape[1]
        if c == 0:
            drafter.seed(0, taps)
            job_taps = None
        else:
            job_taps = taps
        committed = mx.array([anchor], dtype=mx.int32)
        drafter.draft([(None, state, job_taps, committed, None)])
        proposals = mx.reshape(state.drafts, (-1,))
        cohort = drafter._cohort
        slots = drafter.ring_slots
        lo = max(0, total - slots)
        ring_k = [cohort.keys[l][0] for l in range(config.num_hidden_layers)]
        ring_v = [cohort.values[l][0] for l in range(config.num_hidden_layers)]
        pos = np.asarray(cohort.pos[0])
        order = [int(np.where(pos == p)[0][0]) for p in range(lo, total)]
        records.append(
            {
                "taps": rows,
                "anchor": anchor,
                "total": total,
                "positions": np.arange(lo, total, dtype=np.int32),
                "ring_k": [np.asarray(k[:, order, :])[None] for k in ring_k],
                "ring_v": [np.asarray(v[:, order, :])[None] for v in ring_v],
                "hidden": np.asarray(recorded["hidden"]),
                "base_logits": np.asarray(recorded["base_logits"]),
                "step_logits": np.stack([np.asarray(s)[0] for s in steps])[None],
                "tokens": np.asarray(proposals).astype(np.int32),
            }
        )
    return records


COUNTS = [30, 3, 1, 8, 5]
rows = [mx.array(rng.standard_normal((1, n, 4 * H)).astype(np.float32)) for n in COUNTS]
anchors = [int(rng.integers(0, UNPADDED)) for _ in COUNTS]
records = run_cycles(model, rows, anchors)

flat = {"n_cycles": mx.array([len(records)], dtype=mx.int32)}
for c, r in enumerate(records):
    flat[f"{c}.taps"] = r["taps"]
    flat[f"{c}.anchor"] = mx.array([r["anchor"]], dtype=mx.int32)
    flat[f"{c}.total"] = mx.array([r["total"]], dtype=mx.int32)
    flat[f"{c}.positions"] = mx.array(r["positions"])
    for l, (k, v) in enumerate(zip(r["ring_k"], r["ring_v"])):
        flat[f"{c}.ring_k.{l}"] = mx.array(k)
        flat[f"{c}.ring_v.{l}"] = mx.array(v)
    flat[f"{c}.hidden"] = mx.array(r["hidden"])
    flat[f"{c}.base_logits"] = mx.array(r["base_logits"])
    flat[f"{c}.step_logits"] = mx.array(r["step_logits"])
    flat[f"{c}.tokens"] = mx.array(r["tokens"])
mx.save_safetensors(str(out / "reference" / "cycles.safetensors"), flat)

# ---- ablations: how far the drafts move when a mechanism is off --------------------------------
base_tokens = [r["tokens"] for r in records]
base_logits = [r["step_logits"] for r in records]


def moved(label, make):
    """`make()` switches a mechanism off and returns the function that puts it back."""
    restore = make()
    try:
        alt = run_cycles(model, rows, anchors)
    finally:
        restore()
    tokens_changed = sum(int((a["tokens"] != b).sum()) for a, b in zip(alt, base_tokens))
    with np.errstate(invalid="ignore"):  # -inf - -inf past the unpadded vocabulary
        logit_move = max(float(np.abs(a["step_logits"] - b)[np.isfinite(b)].max()) for a, b in zip(alt, base_logits))
    total = sum(len(t) for t in base_tokens)
    print(f"  {label:50s} {tokens_changed:3d} of {total} drafts change, logits move {logit_move:.2e}")
    assert logit_move > 1e-3, f"{label}: the test could not see it"


def set_attr(obj, attr, value):
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    return lambda: setattr(obj, attr, original)


def set_param(path, value):
    """Replace one parameter by name (`layers.0.self_attn.k_proj_ctx.weight`)."""
    original = params[path]
    model.update(tree_unflatten([(path, value)]))
    return lambda: model.update(tree_unflatten([(path, original)]))


def both(*restores):
    return lambda: [r() for r in reversed(restores)]


print("ablations (drafts of 5 cycles x 7):")
uniform = mx.full((4,), 0.25, dtype=mx.bfloat16)
moved("tap mix: every layer averages the taps", lambda: both(
    *[set_attr(l.self_attn, "_alpha", uniform) for l in model.layers]))
moved("tap mix: the Markov head averages the taps", lambda: set_attr(model.markov_head, "_alpha", uniform))
moved("context keys zeroed (layer 1)", lambda: set_param(
    "layers.1.self_attn.k_proj_ctx.weight", mx.zeros_like(params["layers.1.self_attn.k_proj_ctx.weight"])))
moved("context values zeroed (layer 2)", lambda: set_param(
    "layers.2.self_attn.v_proj_ctx.weight", mx.zeros_like(params["layers.2.self_attn.v_proj_ctx.weight"])))
moved("mask embedding zeroed", lambda: both(
    set_param("mask_embed", mx.zeros_like(params["mask_embed"])),
    set_attr(model, "mask_embed", mx.zeros_like(params["mask_embed"]))))
moved("Markov bias folded to zero", lambda: both(
    set_attr(model.markov_head, "_fold_weight", mx.zeros_like(model.markov_head._fold_weight)),
    set_attr(model.markov_head, "_fold_bias", mx.zeros_like(model.markov_head._fold_bias))))
moved("Markov previous-token embedding zeroed", lambda: set_param(
    "markov_head.markov_w1.weight", mx.zeros_like(params["markov_head.markov_w1.weight"])))
print("wrote", out)
