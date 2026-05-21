"""Capture per-input-channel activation statistics for Gemma 4 26B-A4B.

Used as the "importance matrix" input to mixed 3-bit / 4-bit quantization
(see `imatrix_mixed_quant.py`). Idea:

    For each Linear / SwitchLinear in the decoder, accumulate
        H[layer_path][c] = Σ_t (x_t[c])²
    across all calibration tokens t and input channels c.

Weights whose corresponding input column has high H carry more energy through
the matmul, so quantization error in those columns hurts output more. Layers
with broad-spectrum / high-magnitude H are bad candidates for aggressive 3-bit
and should be promoted to 4-bit. This is what V1/V2/V3 predicate experiments
lacked — they used static heuristics; this is data-driven.

Why this is sufficient: per-input-channel sumsq is exactly the diagonal of the
input covariance matrix, scaled by token count. It's the cheapest activation
statistic that captures the magnitude information GPTQ / AWQ / imatrix-style
methods use to drive bit allocation.

Output:
  <out_dir>/importance.npz        — sumsq per path [in_dim]   (bit allocation, imatrix_mixed_quant)
  <out_dir>/activation_stats.npz  — sumabs per path [in_dim]  (AWQ per-column scaling, awq_search)
  <out_dir>/meta.json             — schema_version=2, corpus + token counts metadata

Both sumsq and sumabs come from the same forward pass — capturing both costs
~0 extra time (one extra reduction per Linear call). Files are split so older
consumers reading only `importance.npz` remain working.

Run:
  python scripts/quant/imatrix_capture_gemma4.py \\
    --src /Users/sonheesung/models/gemma-4-26b-a4b-mlx-4bit \\
    --out-dir /Users/sonheesung/models/gemma-4-26b-a4b-imatrix \\
    --max-tokens-per-seq 1024

Why calibrate on the 4-bit build instead of bf16:
  Loading bf16 = 48 GB resident which doesn't fit in 36 GB unified RAM
  (jetsam kills the process mid-forward).  The 4-bit build is 15 GB and
  forwards comfortably.  Quantization noise affects ACTIVATION MAGNITUDES
  only marginally (per-layer cosine ≥ 0.99 vs bf16), and what we save is
  the *relative* per-tensor sensitivity ranking — that ranking is robust
  to small magnitude shifts because we sort tensors by score, not by
  absolute value.
"""

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from imatrix_corpus import build_chat_corpus  # noqa: E402

# Module classes whose `__call__` takes `(self, x)` and we capture sumsq on x[..., -1].
LINEAR_CLASSES = (nn.Linear, nn.QuantizedLinear)
# Module classes whose `__call__` takes `(self, x, indices, sorted_indices=False)`.
SWITCH_CLASSES = (SwitchLinear, QuantizedSwitchLinear)


def install_hooks(model):
    """Monkey-patch Linear/SwitchLinear __call__ to record per-input-channel stats.

    Accumulates two parallel per-input-channel statistics, keyed by module path:
      sumsq[path][c]  = Σ_t (x_t[c])²   — diagonal of input cov; bit-allocation input
      sumabs[path][c] = Σ_t |x_t[c]|    — first-moment magnitude; AWQ scale input

    Both come for free from the same forward pass. Returns
    (sumsq_accum, sumabs_accum, token_counts, uninstall_fn).
    Call uninstall_fn() after forward to restore original methods.
    """
    id_to_path = {}
    for path, m in model.named_modules():
        if isinstance(m, LINEAR_CLASSES + SWITCH_CLASSES):
            id_to_path[id(m)] = path

    sumsq_accum: dict[str, mx.array] = {}
    sumabs_accum: dict[str, mx.array] = {}
    token_counts: dict[str, int] = defaultdict(int)

    # Patch every concrete class — QuantizedLinear has its own __call__
    # separate from nn.Linear so monkey-patching the base alone misses it.
    orig_linear_calls = {cls: cls.__call__ for cls in LINEAR_CLASSES}
    orig_switch_calls = {cls: cls.__call__ for cls in SWITCH_CLASSES}

    def _accum(path: str, x: mx.array) -> None:
        # x shape: [..., in_dim].  Flatten to [N, in_dim], compute sumsq + sumabs → [in_dim].
        x_flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        sumsq = (x_flat * x_flat).sum(axis=0)
        sumabs = mx.abs(x_flat).sum(axis=0)
        if path in sumsq_accum:
            sumsq_accum[path] = sumsq_accum[path] + sumsq
            sumabs_accum[path] = sumabs_accum[path] + sumabs
        else:
            sumsq_accum[path] = sumsq
            sumabs_accum[path] = sumabs
        token_counts[path] += x_flat.shape[0]

    def _make_linear_hook(orig):
        def hook(self, x):
            path = id_to_path.get(id(self))
            if path is not None:
                _accum(path, x)
            return orig(self, x)
        return hook

    def _make_switch_hook(orig):
        def hook(self, x, indices, sorted_indices=False):
            path = id_to_path.get(id(self))
            if path is not None:
                _accum(path, x)
            return orig(self, x, indices, sorted_indices)
        return hook

    for cls, orig in orig_linear_calls.items():
        cls.__call__ = _make_linear_hook(orig)
    for cls, orig in orig_switch_calls.items():
        cls.__call__ = _make_switch_hook(orig)

    def uninstall():
        for cls, orig in orig_linear_calls.items():
            cls.__call__ = orig
        for cls, orig in orig_switch_calls.items():
            cls.__call__ = orig

    return sumsq_accum, sumabs_accum, token_counts, uninstall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="bf16 source model path")
    ap.add_argument("--out-dir", required=True, help="where to write importance.npz")
    ap.add_argument(
        "--max-tokens-per-seq",
        type=int,
        default=1024,
        help="truncate each calibration sequence to this length",
    )
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out_dir = pathlib.Path(args.out_dir)
    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {src}  (forward source — eager) ...")
    t0 = time.time()
    model, tokenizer = load(str(src), tokenizer_config={"eos_token": "<end_of_turn>"})
    print(f"[load] done in {time.time()-t0:.1f}s")

    print("[corpus] building ...")
    sequences = build_chat_corpus(tokenizer, max_tokens_per_seq=args.max_tokens_per_seq)
    print(f"[corpus] {len(sequences)} sequences, "
          f"{sum(len(s) for s in sequences)} total tokens")

    print("[hooks] installing on nn.Linear + SwitchLinear ...")
    sumsq_accum, sumabs_accum, token_counts, uninstall = install_hooks(model)

    print("[forward] running calibration pass ...")
    model.eval()
    total_tokens = 0
    t0 = time.time()
    try:
        for i, ids in enumerate(sequences):
            ids_arr = mx.array(ids)[None]  # [1, T]
            _ = model(ids_arr)
            # Force materialization to free graph before next sequence (memory)
            mx.eval(*list(sumsq_accum.values()), *list(sumabs_accum.values()))
            total_tokens += ids_arr.size
            dt = time.time() - t0
            tps = total_tokens / max(dt, 1e-6)
            print(f"  seq {i+1:3d}/{len(sequences)}  T={ids_arr.size:5d}  "
                  f"cum_tokens={total_tokens:6d}  elapsed={dt:6.1f}s  "
                  f"tok/s={tps:5.1f}")
    finally:
        uninstall()

    print("[save] importance.npz + activation_stats.npz + meta.json ...")
    sumsq_np: dict[str, np.ndarray] = {}
    sumabs_np: dict[str, np.ndarray] = {}
    for path, arr in sumsq_accum.items():
        sumsq_np[path] = np.asarray(arr, dtype=np.float32)
    for path, arr in sumabs_accum.items():
        sumabs_np[path] = np.asarray(arr, dtype=np.float32)
    np.savez_compressed(out_dir / "importance.npz", **sumsq_np)
    np.savez_compressed(out_dir / "activation_stats.npz", **sumabs_np)

    meta = {
        "schema_version": 2,
        "src": str(src),
        "out_dir": str(out_dir),
        "n_sequences": len(sequences),
        "total_tokens": total_tokens,
        "max_tokens_per_seq": args.max_tokens_per_seq,
        "n_tensors": len(sumsq_accum),
        "token_counts_per_path": dict(token_counts),
        "files": {
            "importance.npz": "sumsq per path [in_dim] — diagonal of input cov (bit allocation, imatrix_mixed_quant)",
            "activation_stats.npz": "sumabs per path [in_dim] — first-moment magnitude (AWQ scaling, awq_search)",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(sumsq_accum)} tensors written.  total_tokens={total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
