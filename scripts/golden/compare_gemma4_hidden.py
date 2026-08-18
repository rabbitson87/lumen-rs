#!/usr/bin/env python3
"""Compare MLX (mlx_lm) reference Gemma 4 hidden states against the Rust dump.

Inputs (override via CLI):
    arg1: /tmp/mlx_gemma4_baseline.npz       (from dump_mlx_gemma4_hidden.py)
    arg2: /tmp/rust_gemma4_baseline          (from examples/dump_gemma4_hidden.rs)

Per layer/embed/final_norm/logits: L2-relative error + cosine similarity
between the Rust tensor and the MLX reference. First large jump (cos<0.98 or
L2_rel>0.1) identifies the first diverging layer.
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path

import numpy as np


def read_rust_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"TQHD":
            raise ValueError(f"bad magic {magic!r} in {path}")
        rank = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{rank}I", f.read(rank * 4))
        data = f.read()
    return np.frombuffer(data, dtype=np.float32).reshape(dims)


def compare(ref: np.ndarray, got: np.ndarray, tag: str) -> tuple[float, float]:
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
        f"  {tag:14s}  L2_rel={rel:.4e}  cos={cos:+.6f}  "
        f"max|Δ|={max_abs:.3e}  std(ref/got)=({ref_std:.3f}/{got_std:.3f}){flag}"
    )
    return rel, cos


def main():
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mlx_gemma4_baseline.npz")
    rust_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/rust_gemma4_baseline")
    # Optional 3rd arg: Rust filename prefix (e.g. "step12_") for decode-step dumps.
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""

    print(f"MLX  ref : {npz_path}")
    print(f"Rust dir : {rust_dir}  prefix={prefix!r}\n")

    mlx = np.load(str(npz_path))
    order = ["embed"] + [f"L{i:02d}" for i in range(30)] + ["final_norm", "logits_last"]

    for tag in order:
        if tag == "logits_last":
            rust_path = rust_dir / f"{prefix}logits.bin"
        else:
            rust_path = rust_dir / f"{prefix}{tag}.bin"
        if not rust_path.exists():
            print(f"  {tag:14s}  missing {rust_path}")
            continue
        rust = read_rust_bin(rust_path)
        ref = mlx[tag]
        if tag == "logits_last":
            # Rust dumps full [B=1, S, V]; MLX stores only the last row.
            rust_last = rust[0, -1] if rust.ndim == 3 else rust[-1]
            compare(ref, rust_last, tag)
        else:
            compare(ref, rust, tag)

    # Top-5 argmax (last token)
    rust_logits = read_rust_bin(rust_dir / f"{prefix}logits.bin")
    rust_last = rust_logits[0, -1] if rust_logits.ndim == 3 else rust_logits[-1]
    rust_top = np.argsort(rust_last)[-5:][::-1]
    mlx_last = mlx["logits_last"]
    mlx_top = np.argsort(mlx_last)[-5:][::-1]
    print("\nRust top-5 last-token:")
    for t in rust_top.tolist():
        print(f"  {t:>6}  logit={rust_last[t]:+.4f}")
    print("MLX  top-5 last-token:")
    for t in mlx_top.tolist():
        print(f"  {t:>6}  logit={mlx_last[t]:+.4f}")
    print(f"\nMLX recorded argmax : {int(mlx['argmax'][0])}")
    print(f"Rust argmax (last)  : {int(rust_top[0])}")


if __name__ == "__main__":
    main()
