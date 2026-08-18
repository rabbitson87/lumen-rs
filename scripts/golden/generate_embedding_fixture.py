#!/usr/bin/env python3
"""Generate self-contained embedding-lookup fixtures for the mlx-rs FFI parity
test under `crates/lumen-mlx/src/native_embedding.rs`.

Two fixtures are emitted:

* `embed_plain_<name>.bin` — plain f32 weight + uint32 token ids + expected f32
  output (`mx.take(weight, ids, axis=0)`).
* `embed_mxfp4_<name>.bin` — MXFP4-quantized weight (packed u32 + scales u8) +
  uint32 token ids + expected f32 output, where the reference is produced by
  `mx.dequantize(...mode="mxfp4")` *after* row-selection (matches the
  Phase 3b.1 production-style quantized embedding lookup pattern used by
  `nn.QuantizedEmbedding.forward` upstream).

File formats (little-endian):

    embed_plain header (magic b"TQEP" + 4 u32 fields):
        magic          4s
        vocab_size     u32
        hidden_size    u32
        n_tokens       u32
        weight_dtype   u32   (0 = f32; reserved for future bf16/f16)
        weight  f32[vocab*hidden]    row-major
        ids     u32[n_tokens]
        ref_y   f32[n_tokens*hidden] row-major

    embed_mxfp4 header (magic b"TQEM" + 6 u32 fields):
        magic          4s
        vocab_size     u32
        hidden_size    u32
        n_tokens       u32
        group_size     u32   (32 for mxfp4)
        bits           u32   (4 for mxfp4)
        packed_len_u32 u32   = vocab_size * hidden_size / 8
        scales_len     u32   = vocab_size * hidden_size / 32
        packed   u32[packed_len_u32]   row-major [vocab, hidden/8]
        scales   u8 [scales_len]       row-major [vocab, hidden/32]
        ids      u32[n_tokens]
        ref_y    f32[n_tokens*hidden]  row-major

Both formats keep token ids and the reference output (after dequant + cast to
f32 to match the Rust read path) so the parity test can do bit-identical
comparison without touching MLX directly.

Requires `mlx` (Apple Silicon only).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import numpy as np
except ImportError as exc:
    print(f"error: missing dependency ({exc}). install with: pip install mlx numpy")
    sys.exit(1)

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "lumen-metal"
    / "tests"
    / "fixtures"
)
GROUP_SIZE = 32
BITS = 4
PLAIN_MAGIC = b"TQEP"
MXFP4_MAGIC = b"TQEM"


def write_plain_fixture(name: str, vocab: int, hidden: int, n_tokens: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    weight_f32 = rng.normal(loc=0.0, scale=1.0, size=(vocab, hidden)).astype(np.float32)
    ids = rng.integers(low=0, high=vocab, size=(n_tokens,), dtype=np.uint32)

    weight_mx = mx.array(weight_f32)
    ids_mx = mx.array(ids.astype(np.int32))  # mlx take wants int indices

    out_mx = mx.take(weight_mx, ids_mx, axis=0).astype(mx.float32)
    mx.eval(out_mx)
    out_np = np.array(out_mx, copy=False).astype(np.float32)

    assert out_np.shape == (n_tokens, hidden), f"plain: unexpected ref shape {out_np.shape}"

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"embed_plain_{name}.bin"
    with open(path, "wb") as f:
        f.write(PLAIN_MAGIC)
        f.write(struct.pack("<IIII", vocab, hidden, n_tokens, 0))
        f.write(weight_f32.tobytes())
        f.write(ids.tobytes())
        f.write(out_np.tobytes())

    print(
        f"wrote {path.name}: vocab={vocab} hidden={hidden} n_tokens={n_tokens} "
        f"weight={weight_f32.nbytes}B ids={ids.nbytes}B ref={out_np.nbytes}B"
    )


def write_mxfp4_fixture(name: str, vocab: int, hidden: int, n_tokens: int, seed: int) -> None:
    assert hidden % GROUP_SIZE == 0, f"hidden {hidden} must be a multiple of group_size {GROUP_SIZE}"

    rng = np.random.default_rng(seed)
    weight_f32 = rng.normal(loc=0.0, scale=1.0, size=(vocab, hidden)).astype(np.float32)
    ids = rng.integers(low=0, high=vocab, size=(n_tokens,), dtype=np.uint32)

    weight_mx = mx.array(weight_f32)
    result = mx.quantize(weight_mx, group_size=GROUP_SIZE, bits=BITS, mode="mxfp4")
    if len(result) == 2:
        packed, scales = result
    elif len(result) == 3 and result[2] is None:
        packed, scales, _ = result
    else:
        raise RuntimeError(f"unexpected mxfp4 quantize return shape: {len(result)} items")

    packed_u32 = packed.view(mx.uint32)
    scales_u8 = scales.view(mx.uint8)

    ids_mx = mx.array(ids.astype(np.int32))

    # Mirror nn.QuantizedEmbedding.forward: select rows from packed + scales,
    # then dequantize the small selected slice with mode="mxfp4".
    selected_packed = mx.take(packed_u32, ids_mx, axis=0)
    selected_scales = mx.take(scales_u8, ids_mx, axis=0)
    out_bf16 = mx.dequantize(
        selected_packed,
        selected_scales,
        biases=None,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="mxfp4",
    )
    out_f32 = out_bf16.astype(mx.float32)

    mx.eval(packed_u32, scales_u8, out_f32)

    packed_np = np.array(packed_u32, copy=False).astype(np.uint32)
    scales_np = np.array(scales_u8, copy=False).astype(np.uint8)
    out_np = np.array(out_f32, copy=False).astype(np.float32)

    expected_packed_len = vocab * hidden // 8
    expected_scales_len = vocab * hidden // 32
    assert packed_np.size == expected_packed_len, (
        f"packed size {packed_np.size} != expected {expected_packed_len}"
    )
    assert scales_np.size == expected_scales_len, (
        f"scales size {scales_np.size} != expected {expected_scales_len}"
    )
    assert out_np.shape == (n_tokens, hidden), f"mxfp4: unexpected ref shape {out_np.shape}"

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"embed_mxfp4_{name}.bin"
    with open(path, "wb") as f:
        f.write(MXFP4_MAGIC)
        f.write(
            struct.pack(
                "<IIIIIII",
                vocab,
                hidden,
                n_tokens,
                GROUP_SIZE,
                BITS,
                packed_np.size,
                scales_np.size,
            )
        )
        f.write(packed_np.tobytes())
        f.write(scales_np.tobytes())
        f.write(ids.tobytes())
        f.write(out_np.tobytes())

    print(
        f"wrote {path.name}: vocab={vocab} hidden={hidden} n_tokens={n_tokens} "
        f"packed={packed_np.nbytes}B scales={scales_np.nbytes}B ids={ids.nbytes}B "
        f"ref={out_np.nbytes}B"
    )


def main() -> None:
    # Plain f32 embedding lookups: small + multi-token shapes.
    write_plain_fixture("tiny", vocab=16, hidden=32, n_tokens=4, seed=0xE17)
    write_plain_fixture("small", vocab=128, hidden=128, n_tokens=8, seed=0xE12)

    # MXFP4-quantized embedding (matches Qwen3.6-35B-A3B-mxfp4's
    # `embed_tokens` quantization scheme: group_size=32, bits=4, mode="mxfp4").
    write_mxfp4_fixture("tiny", vocab=16, hidden=32, n_tokens=4, seed=0xE71)
    write_mxfp4_fixture("small", vocab=64, hidden=128, n_tokens=8, seed=0xE73)

    print(f"embedding fixtures written under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
