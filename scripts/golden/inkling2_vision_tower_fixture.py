#!/usr/bin/env python3
"""Tiny random-weights replica of Inkling 2's vision tower, run by the oMLX reference.

usage: python scripts/golden/inkling2_vision_tower_fixture.py <dir containing omlx/patches/inkling2> \\
           crates/lumen-mlx/tests/fixtures/inkling2_vision/tower

The reference is `omlx/patches/inkling2/vision.py` (oMLX 0.8.2, Apache-2.0), imported by path; it
needs only mlx and numpy. Writes into the output directory:

  config.json            `vision_config` of a checkpoint of the same schema, tiny sizes
  weights.safetensors    the tower's tensors under the checkpoint's names (`vision_tower.*`), float32
  reference/inputs.safetensors   `image_<k>`: the pixel blocks of three images of different shapes,
                         `[rows * cols, block, block, 3]` float32 (values in [-1, 1] like a
                         normalized image, with a few of the processor's padding value)
  reference/outputs.safetensors  `image_<k>.features` (the reference's `[rows * cols, text_hidden]`) and
                         `image_<k>.stage_<n>`: the tower's state after the embeddings (class token
                         first), after `ln_pre`, and after every encoder layer

The shape mirrors the real tower (class token, LayerScale, 2 x 2 merge, a position table that is
resampled to each image); only the widths and depth are small. The table is 5 x 5 so that the
images' patch grids (2 x 2, 4 x 6, 6 x 2) are resampled both up and down. The script prints how far
the features move when each mechanism is switched off, so a mechanism the test cannot see is
visible here. Seeds are fixed; mlx 0.31.2 produced the committed files.
"""
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

ref_root, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "inkling2_vision", ref_root / "omlx" / "patches" / "inkling2" / "vision.py"
)
vision = importlib.util.module_from_spec(spec)
sys.modules["inkling2_vision"] = vision  # `@dataclass` looks its module up here
spec.loader.exec_module(vision)

rng = np.random.default_rng(23)
CONFIG = {
    "model_type": "inkling2_vision",
    "text_hidden_size": 20,
    "temporal_patch_size": 2,
    "native_patch_size": 4,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 24,
    "projector_bias": False,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 5,
    "layer_norm_eps": 1e-05,
}
args = vision.VisionArgs.from_dict(CONFIG)
tower = vision.VisionTower(args, CONFIG["text_hidden_size"])

# Random weights, shaped like trained ones: LayerScale gammas near 1 (so every layer matters),
# norms near the identity, small linears.
params = {}
for name, value in tree_flatten(tower.parameters()):
    shape = value.shape
    if name.endswith("gamma"):
        data = rng.uniform(0.5, 1.5, shape)
    elif name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_pre.weight"):
        data = 1.0 + 0.1 * rng.standard_normal(shape)
    elif name.endswith(".bias"):
        data = 0.1 * rng.standard_normal(shape)
    elif name.endswith("embedding") or name.endswith("class_embedding"):
        data = 0.5 * rng.standard_normal(shape)
    else:
        data = rng.standard_normal(shape) / np.sqrt(shape[-1])
    params[name] = mx.array(data.astype(np.float32))
tower.update(tree_unflatten(list(params.items())))
mx.eval(tower.parameters())

block = args.patch_size * args.spatial_merge_size
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps({"vision_config": CONFIG}, indent=1) + "\n")
mx.save_safetensors(
    str(out / "weights.safetensors"), {f"vision_tower.{k}": v for k, v in params.items()}
)

SHAPES = [(1, 1), (2, 3), (3, 1)]  # (rows, cols) of blocks
PAD = (-1.0 * 0.00392156862745098 - 0.5) / 0.5


def pixels(rows, cols):
    x = rng.uniform(-1.0, 1.0, (rows * cols, block, block, 3)).astype(np.float32)
    x[-1, block // 2 :, block // 2 :, :] = np.float32(PAD)  # a padded corner, as a processor leaves
    return x


def run(tower, blocks, rows, cols, stages=None):
    """`VisionTower.__call__` for one image, optionally keeping the stages."""
    emb = tower.backbone.embeddings
    h = emb(blocks, rows, cols)
    if stages is not None:
        stages.append(h)
    h = tower.backbone.ln_pre(h)
    if stages is not None:
        stages.append(h)
    cos, sin = vision._axial_rope(rows, cols, tower.args)
    for layer in tower.backbone.layers:
        h = layer(h, cos, sin)
        if stages is not None:
            stages.append(h)
    count = rows * cols
    patches = h[1:].reshape(count, args.spatial_merge_size**2 * h.shape[-1])
    return tower.proj_out(vision.nn.gelu(tower.proj_in(patches)))


inputs, outputs = {}, {}
images = []
for k, (rows, cols) in enumerate(SHAPES):
    x = mx.array(pixels(rows, cols))
    stages = []
    features = run(tower, x, rows, cols, stages)
    # The reference's own entry point must agree with the staged copy.
    whole = tower(x, np.array([[1, rows, cols]]))
    assert mx.array_equal(whole, features).item(), "the staged run drifted from VisionTower.__call__"
    mx.eval(features, stages)
    inputs[f"image_{k}"] = x
    outputs[f"image_{k}.features"] = features
    for n, s in enumerate(stages):
        outputs[f"image_{k}.stage_{n}"] = s
    images.append((rows, cols, x, features))
(out / "reference").mkdir(exist_ok=True)
mx.save_safetensors(str(out / "reference" / "inputs.safetensors"), inputs)
mx.save_safetensors(str(out / "reference" / "outputs.safetensors"), outputs)

# Ablations: how far the features move when a mechanism is off. A zero would hide it from the test.
rows, cols, x, base = images[1]
scale = float(mx.max(mx.abs(base)))


def moved(features):
    return float(mx.max(mx.abs(features - base))) / scale


def with_patch(attr_owner, attr, replacement):
    original = getattr(attr_owner, attr)
    setattr(attr_owner, attr, replacement)
    try:
        return run(tower, x, rows, cols)
    finally:
        setattr(attr_owner, attr, original)


table = {}
emb = tower.backbone.embeddings
saved_pos = emb.position_embedding
emb.position_embedding = mx.zeros_like(saved_pos)
table["position table zeroed"] = moved(run(tower, x, rows, cols))
emb.position_embedding = saved_pos

original_rope = vision._axial_rope
vision._axial_rope = lambda r, c, a: (
    mx.ones_like(original_rope(r, c, a)[0]),
    mx.zeros_like(original_rope(r, c, a)[1]),
)
table["no rotary embedding"] = moved(run(tower, x, rows, cols))


def swapped(r, c, a):
    cos, sin = original_rope(r, c, a)
    half = cos.shape[-1] // 2
    return (
        mx.concatenate([cos[:, half:], cos[:, :half]], axis=-1),
        mx.concatenate([sin[:, half:], sin[:, :half]], axis=-1),
    )


vision._axial_rope = swapped
table["width and height frequencies swapped"] = moved(run(tower, x, rows, cols))
vision._axial_rope = original_rope

original_coords = vision._grid_coords
vision._grid_coords = lambda r, c, m: (
    np.arange(r * c) // c,
    np.arange(r * c) % c,
)
table["patches in raster order, not block order"] = moved(run(tower, x, rows, cols))
vision._grid_coords = original_coords

gammas = [(l.ls_1.gamma, l.ls_2.gamma) for l in tower.backbone.layers]
for l in tower.backbone.layers:
    l.ls_1.gamma, l.ls_2.gamma = mx.ones_like(l.ls_1.gamma), mx.ones_like(l.ls_2.gamma)
table["layer scale = 1"] = moved(run(tower, x, rows, cols))
for l, (g1, g2) in zip(tower.backbone.layers, gammas):
    l.ls_1.gamma, l.ls_2.gamma = g1, g2

cls_saved = emb.class_embedding
# Not a constant shift: ln_pre removes a mean shift, so +1.0 would move nothing.
emb.class_embedding = cls_saved + mx.array(rng.standard_normal(cls_saved.shape).astype(np.float32))
table["class token perturbed"] = moved(run(tower, x, rows, cols))
emb.class_embedding = cls_saved

assert abs(moved(run(tower, x, rows, cols))) == 0.0, "the ablations did not restore the tower"
print("features move by (relative to the largest feature) when:")
for name, delta in table.items():
    print(f"  {name:44s} {delta:.3e}")
    assert delta > 1e-2, f"{name}: the test could not see it"
print("wrote", out)
