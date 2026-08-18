#!/usr/bin/env python3
"""Dump a single int8-affine weight (MoE gate) plus MLX's dequantized reference
so the Rust `dequant_int8_affine` can be compared element-wise.

Writes `/tmp/int8_affine_ref.bin`:
    magic       4s    b"TQI8"
    out_dim     u32
    in_dim      u32
    group_size  u32   (64 for all shipped int8-affine overrides)
    bits        u32   (8)
    packed_len  u32   (= out_dim * in_dim / 4)   u32 values
    scales_len  u32   (= out_dim * in_dim / group_size)  bf16 u16
    biases_len  u32   (= scales_len)              bf16 u16
    ref_len     u32   (= out_dim * in_dim)        f32
    packed      u32[packed_len]
    scales      u16[scales_len]   (raw bf16 bits, little-endian)
    biases      u16[biases_len]
    expected_w  f32[ref_len]       (row-major [out_dim, in_dim], MLX dequantized)
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np
import mlx.core as mx


def main() -> None:
    snap = Path(
        "/path/to/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/"
        "833013b27a1f7c6dbb008b55d37c387ea22ea89d"
    )

    # Layer 0 MoE gate: int8-affine, bits=8, group_size=64.
    prefix = "language_model.model.layers.0.mlp.gate"

    # Find which shard holds it.
    import json
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shard_w = snap / wmap[f"{prefix}.weight"]
    shard_s = snap / wmap[f"{prefix}.scales"]
    shard_b = snap / wmap[f"{prefix}.biases"]

    packed = mx.load(str(shard_w))[f"{prefix}.weight"]       # uint32, (256, 512)
    scales = mx.load(str(shard_s))[f"{prefix}.scales"]        # bf16, (256, 32)
    biases = mx.load(str(shard_b))[f"{prefix}.biases"]        # bf16, (256, 32)

    print(f"packed: {packed.dtype} {packed.shape}")
    print(f"scales: {scales.dtype} {scales.shape}")
    print(f"biases: {biases.dtype} {biases.shape}")

    out_dim, packed_cols = packed.shape
    in_dim = packed_cols * 4   # 4 uint8 per uint32
    group_size = in_dim * out_dim // (scales.shape[0] * scales.shape[1])
    assert group_size == 64, f"unexpected group_size {group_size}"

    # MLX reference dequant.
    w_ref = mx.dequantize(packed, scales, biases, group_size=64, bits=8)
    mx.eval(w_ref)
    w_ref_np = np.asarray(w_ref.astype(mx.float32))  # (256, 2048)
    print(f"w_ref: {w_ref_np.dtype} {w_ref_np.shape}")
    print(f"  min={w_ref_np.min():+.4f}  max={w_ref_np.max():+.4f}  "
          f"mean={w_ref_np.mean():+.4f}  std={w_ref_np.std():+.4f}")

    # Raw byte views for writing.
    packed_np = np.asarray(packed)  # uint32
    # bf16 → raw u16 bits
    def bf16_bits(t: mx.array) -> np.ndarray:
        # view as uint16; MLX has no direct bitcast helper, but np.frombuffer works on
        # the MLX buffer bytes.
        b = bytes(memoryview(t))
        return np.frombuffer(b, dtype=np.uint16)
    scales_bits = bf16_bits(scales)
    biases_bits = bf16_bits(biases)

    out_path = Path("/tmp/int8_affine_ref.bin")
    with open(out_path, "wb") as f:
        f.write(b"TQI8")
        f.write(struct.pack(
            "<IIIIIIII",
            out_dim,
            in_dim,
            group_size,
            8,
            packed_np.size,
            scales_bits.size,
            biases_bits.size,
            w_ref_np.size,
        ))
        f.write(packed_np.astype(np.uint32, copy=False).tobytes())
        f.write(scales_bits.astype(np.uint16, copy=False).tobytes())
        f.write(biases_bits.astype(np.uint16, copy=False).tobytes())
        f.write(w_ref_np.astype(np.float32, copy=False).tobytes())
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
