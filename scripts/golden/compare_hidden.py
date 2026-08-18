#!/usr/bin/env python3
"""Compare MLX reference hidden states against the Rust dump.

Inputs:
    /tmp/mlx_hidden.npz      (from scripts/dump_mlx_hidden.py)
    /tmp/rust_hidden/*.bin   (from the dump_qwen35_hidden example binary)

Prints, per layer/embed/final/logits, L2-relative error + cosine similarity
between the Rust tensor and the MLX reference. The first large jump identifies
the first diverging layer — anything with cosine < 0.98 or L2 rel > 0.1 is worth
investigating.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np


RUST_DIR = Path("/tmp/rust_hidden")


def read_rust_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"TQHD", f"bad magic in {path}"
        rank = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{rank}I", f.read(rank * 4))
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float32).reshape(dims)
    return arr


def compare(ref: np.ndarray, got: np.ndarray, tag: str) -> tuple[float, float]:
    # Align shapes: Rust dumps [B=1, S, H] → drop batch. MLX ref is stored
    # after a [0] index so it's [S, H] (or [S, vocab] or [S, hidden]).
    if got.ndim == 3 and got.shape[0] == 1:
        got = got[0]
    if ref.shape != got.shape:
        print(f"  {tag:14s}  SHAPE MISMATCH  ref={ref.shape} got={got.shape}")
        return float("nan"), float("nan")
    diff = got - ref
    l2_err = float(np.sqrt((diff * diff).sum()))
    l2_ref = float(np.sqrt((ref * ref).sum()) + 1e-12)
    rel = l2_err / l2_ref
    dot = float((got * ref).sum())
    ng = float(np.sqrt((got * got).sum()) + 1e-12)
    nr = float(np.sqrt((ref * ref).sum()) + 1e-12)
    cos = dot / (ng * nr)
    max_abs = float(np.max(np.abs(diff)))
    ref_std = float(ref.std())
    got_std = float(got.std())
    flag = "" if cos > 0.98 else "  *** DIVERGE"
    print(
        f"  {tag:14s}  "
        f"L2_rel={rel:.4e}  cos={cos:+.6f}  "
        f"max|Δ|={max_abs:.3e}  "
        f"std(ref/got)=({ref_std:.3f}/{got_std:.3f}){flag}"
    )
    return rel, cos


def main():
    mlx = np.load("/tmp/mlx_hidden.npz")
    print(f"MLX archive keys: {sorted(mlx.keys())[:6]}...")

    order = ["embed"] + [f"L{i:02d}" for i in range(40)] + ["final_norm", "logits_last"]

    for tag in order:
        if tag == "logits_last":
            rust_path = RUST_DIR / "logits.bin"
        else:
            rust_path = RUST_DIR / f"{tag}.bin"
        if not rust_path.exists():
            print(f"  {tag:14s}  missing {rust_path}")
            continue
        rust = read_rust_bin(rust_path)
        ref = mlx[tag]
        if tag == "logits_last":
            # Rust dumps full [S, vocab]; MLX stores only the last row.
            rust_last = rust[0, -1] if rust.ndim == 3 else rust[-1]
            compare(ref, rust_last, tag)
        else:
            compare(ref, rust, tag)

    # Also dump top-5 argmax from Rust logits and MLX's recorded argmax.
    rust_logits_full = read_rust_bin(RUST_DIR / "logits.bin")
    rust_last = rust_logits_full[0, -1] if rust_logits_full.ndim == 3 else rust_logits_full[-1]
    rust_top = np.argsort(rust_last)[-5:][::-1]
    print("\nRust top-5 last-token logits:")
    for t in rust_top.tolist():
        print(f"  {t}: logit={rust_last[t]:+.4f}")
    print(f"\nMLX recorded argmax: {int(mlx['argmax'][0])}")


if __name__ == "__main__":
    main()
