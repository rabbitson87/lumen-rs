#!/usr/bin/env python3
"""Golden data for Inkling 2's image path, from the oMLX reference.

usage: python scripts/golden/inkling2_vision_golden.py <dir containing omlx/patches/inkling2> \\
           crates/lumen-mlx/tests/fixtures/inkling2_vision

Writes into the output directory:

  *.png            test images, one per colour type the processor treats differently
  *.blocks.bin     the reference processor's pixel blocks for each, float32 little-endian,
                   `[rows * cols, 40, 40, 3]`
  pixels.json      the index: name, png, blocks, rows, cols

The Rust test (`src/inkling2/image.rs`) decodes each PNG with its own decoder and has to produce
the same blocks bit for bit. PNG only: JPEG decoders disagree about the last bit of a pixel, which
would be a test of the decoder and not of the processor.

The reference is `omlx/patches/inkling2/processing.py` (oMLX 0.8.2, Apache-2.0), imported by path;
it needs transformers (for its tokenizer class, unused here) and Pillow. The committed files were
produced with Pillow 12.2.0, numpy 2.x.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ref_root, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "inkling2_processing", ref_root / "omlx" / "patches" / "inkling2" / "processing.py"
)
processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing)
# The values of the checkpoint's processor_config.json.
processor = processing.Inkling2ImageProcessor(
    size={"height": 40, "width": 40},
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    rescale_factor=0.00392156862745098,
)

rng = np.random.default_rng(5)
out.mkdir(parents=True, exist_ok=True)


def noise(h, w, c):
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


images = {}
images["rgb 37x53 (one column, two rows of blocks)"] = Image.fromarray(noise(53, 37, 3), "RGB")
rgba = noise(41, 45, 4)
rgba[..., 3] = np.linspace(0, 255, 41 * 45).reshape(41, 45).astype(np.uint8)
images["rgba 45x41 with an alpha ramp"] = Image.fromarray(rgba, "RGBA")
images["gray 40x40 (exactly one block)"] = Image.fromarray(noise(40, 40, 1)[..., 0], "L")
la = noise(23, 17, 2)
images["gray with alpha 17x23"] = Image.fromarray(la, "LA")
palette = Image.fromarray(noise(30, 50, 1)[..., 0] % 8, "P")
palette.putpalette([int(v) for v in rng.integers(0, 256, size=24)] + [0] * (768 - 24))
palette.info["transparency"] = 3
images["palette with a transparent index 50x30"] = palette
images["rgb 80x40 (two blocks across)"] = Image.fromarray(noise(40, 80, 3), "RGB")
opaque = noise(20, 20, 4)
opaque[..., 3] = 255
images["rgba 20x20, fully opaque"] = Image.fromarray(opaque, "RGBA")
images["rgb 1x1"] = Image.fromarray(noise(1, 1, 3), "RGB")
images["rgb 41x81 (one pixel over a block each way)"] = Image.fromarray(noise(81, 41, 3), "RGB")

index = []
for n, (name, image) in enumerate(images.items()):
    stem = f"image_{n:02d}"
    png = out / f"{stem}.png"
    image.save(png)
    # The reference reads the file back, as a server would, so a palette or a colour key is
    # whatever the file says, not what the array in memory said.
    loaded = Image.open(png)
    loaded.load()
    result = processor.preprocess([loaded])
    blocks = np.ascontiguousarray(result["pixel_values"], dtype="<f4")
    _, rows, cols = result["image_grid_thw"][0].tolist()
    assert blocks.shape == (rows * cols, 40, 40, 3), blocks.shape
    (out / f"{stem}.blocks.bin").write_bytes(blocks.tobytes())
    index.append(
        {
            "name": name,
            "png": png.name,
            "blocks": f"{stem}.blocks.bin",
            "rows": rows,
            "cols": cols,
            "mode": loaded.mode,
            "transparent": bool(loaded.has_transparency_data),
        }
    )
    print(f"{name:48s} {loaded.mode:4s} transparent={loaded.has_transparency_data!s:5s} {rows}x{cols} blocks")

(out / "pixels.json").write_text(json.dumps({"images": index}, indent=1) + "\n")
print("wrote", out)
