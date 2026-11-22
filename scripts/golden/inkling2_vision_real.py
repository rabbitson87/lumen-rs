#!/usr/bin/env python3
"""The reference's image features on the real Inkling-2-Mini checkpoint, for one scene.

usage: python scripts/golden/inkling2_vision_real.py <dir containing omlx/patches/inkling2> \\
           <checkpoint dir> <out dir>

Writes into <out dir>:
  scene.png                 a 200 x 120 drawing (a red square, a blue circle), deterministic
  scene_alt.png             the same size, another picture (a yellow circle, a green square)
  features_bf16.safetensors the reference tower's output on the checkpoint's own (bfloat16) weights
  features_f32.safetensors  the same tower with every weight widened to float32: the answer the
                            bfloat16 run is an approximation of, and so the noise floor to read a
                            bfloat16 comparison against

`crates/lumen-mlx/examples/inkling2_vision.rs` compares the Rust tower with both. The checkpoint is
not in the repository, so neither is the output; the two scenes are also committed under
`tests/fixtures/inkling2_vision/` for the ignored real-checkpoint tests.
"""
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageDraw

ref_root, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"inkling2_{name}", ref_root / "omlx" / "patches" / "inkling2" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vision, processing = load("vision"), load("processing")

scene = Image.new("RGB", (200, 120), (255, 255, 255))
draw = ImageDraw.Draw(scene)
draw.rectangle([20, 25, 90, 95], fill=(220, 30, 30))
draw.ellipse([115, 25, 185, 95], fill=(30, 60, 220))
scene.save(out / "scene.png")
scene = Image.open(out / "scene.png")
scene.load()

# The same size with another picture (a green square, a yellow circle, swapped sides): same token
# grid, different content, for the test that a retained conversation is not reused across them.
alt = Image.new("RGB", (200, 120), (255, 255, 255))
draw = ImageDraw.Draw(alt)
draw.ellipse([15, 25, 85, 95], fill=(240, 200, 20))
draw.rectangle([115, 25, 185, 95], fill=(30, 160, 60))
alt.save(out / "scene_alt.png")

processor = processing.Inkling2ImageProcessor(
    size={"height": 40, "width": 40},
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    rescale_factor=0.00392156862745098,
)
result = processor.preprocess([scene])
blocks = mx.array(np.ascontiguousarray(result["pixel_values"], dtype=np.float32))
grid = result["image_grid_thw"]
print("grid", np.asarray(grid).tolist(), "blocks", blocks.shape)

config = json.loads((ckpt / "config.json").read_text())["vision_config"]
weights = {}
for shard in sorted(ckpt.glob("*.safetensors")):
    for key, value in mx.load(str(shard)).items():
        if key.startswith("vision_tower."):
            weights[key.removeprefix("vision_tower.")] = value


def run(dtype):
    tower = vision.VisionTower(vision.VisionArgs.from_dict(config), config["text_hidden_size"])
    tower.load_weights([(k, v.astype(dtype)) for k, v in weights.items()])
    mx.eval(tower.parameters())
    features = tower(blocks, grid)
    mx.eval(features)
    return features


bf16, f32 = run(mx.bfloat16), run(mx.float32)
mx.save_safetensors(str(out / "features_bf16.safetensors"), {"features": bf16.astype(mx.float32)})
mx.save_safetensors(str(out / "features_f32.safetensors"), {"features": f32})
rows = np.asarray(bf16.astype(mx.float32))
exact = np.asarray(f32)
cos = (rows * exact).sum(-1) / (np.linalg.norm(rows, axis=-1) * np.linalg.norm(exact, axis=-1))
print("features", f32.shape, "bf16 vs f32 reference: min cosine", float(cos.min()),
      "max |diff|", float(np.abs(rows - exact).max()), "of", float(np.abs(exact).max()))
