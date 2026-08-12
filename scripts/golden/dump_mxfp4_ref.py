#!/usr/bin/env python3
"""Dump a single MXFP4 layer's packed weights + scales + a fixed activation and
the MLX reference matmul output to a binary file that the Rust parity test
consumes.

Output format (little-endian):
    magic            4s         b"TQMM"
    out_features     u32
    in_features      u32
    group_size       u32        (32 for mxfp4)
    bits             u32        (4 for mxfp4)
    x_len            u32        = in_features
    packed_len_u32   u32        = out_features * in_features / 8
    scales_len       u32        = out_features * in_features / 32
    y_len            u32        = out_features
    x                f32[x_len]              (row-major, 1 batch)
    packed           u32[packed_len_u32]     (row-major, [out, in/8])
    scales           u8[scales_len]          (row-major, [out, in/32])
    expected_y       f32[y_len]              (MLX mx.quantized_matmul output)

The weight is `language_model.model.layers.3.self_attn.q_proj` because that is
the first full-attention layer (pattern: linear×3 + full×1); it has
shape (8192, 2048) unpacked, i.e. gated-Q output layout.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open

import json


def main():
    snap = Path(
        "/path/to/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/"
        "833013b27a1f7c6dbb008b55d37c387ea22ea89d"
    )
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]

    key_w = "language_model.model.layers.3.self_attn.q_proj.weight"
    key_s = "language_model.model.layers.3.self_attn.q_proj.scales"

    # Read raw tensors via safetensors (numpy view).
    shard_w = snap / wmap[key_w]
    shard_s = snap / wmap[key_s]
    with safe_open(str(shard_w), framework="numpy") as f:
        packed_np = f.get_tensor(key_w)  # dtype uint32, shape (out, in/8)
    with safe_open(str(shard_s), framework="numpy") as f:
        scales_np = f.get_tensor(key_s)  # dtype uint8, shape (out, in/32)

    out_features, packed_cols = packed_np.shape
    in_features = packed_cols * 8
    print(f"q_proj: out={out_features}, in={in_features}")
    print(f"  packed: {packed_np.shape} {packed_np.dtype}")
    print(f"  scales: {scales_np.shape} {scales_np.dtype}")

    # Fixed activation (seed=42).
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(in_features, dtype=np.float32) * 0.1

    # MLX reference via mx.quantized_matmul.
    # mx.quantized_matmul expects `w` as the packed uint32 with shape (out, in/8),
    # `scales` as per-group scalar (uint8 for E8M0). Mode "mxfp4" → E2M1 decode.
    x_mx = mx.array(x_np)  # (in,)
    w_mx = mx.array(packed_np)
    s_mx = mx.array(scales_np)

    # Simulate the forward: y = x @ W^T where W is unpacked (out, in).
    # MLX's quantized_matmul has `transpose=True` for this "W(out,in) * x(...,in)" pattern.
    y_mx = mx.quantized_matmul(
        x_mx,
        w_mx,
        s_mx,
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    mx.eval(y_mx)
    y_np = np.asarray(y_mx, dtype=np.float32)
    print(f"  y_ref: shape={y_np.shape}, dtype={y_np.dtype}")
    print(f"  y stats: min={y_np.min():.4f} max={y_np.max():.4f} "
          f"mean={y_np.mean():.4f} std={y_np.std():.4f}")

    # ── Write binary file ──────────────────────────────────────────────────
    out_path = Path("/tmp/mxfp4_ref.bin")
    with open(out_path, "wb") as f:
        f.write(b"TQMM")
        f.write(struct.pack(
            "<IIIIIIII",
            out_features,
            in_features,
            32,                        # group_size
            4,                         # bits
            x_np.size,                 # x_len
            packed_np.size,            # packed_len_u32
            scales_np.size,            # scales_len (bytes)
            y_np.size,                 # y_len
        ))
        f.write(x_np.tobytes())
        f.write(packed_np.astype(np.uint32, copy=False).tobytes())
        f.write(scales_np.astype(np.uint8, copy=False).tobytes())
        f.write(y_np.tobytes())
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
