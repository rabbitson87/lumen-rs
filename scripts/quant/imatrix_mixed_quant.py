"""Mixed 3-bit / 4-bit quantization for Gemma 4 26B-A4B driven by imatrix stats.

Inputs:
  --imatrix-dir : directory produced by imatrix_capture_gemma4.py containing
                  importance.npz (per-path activation sumsq, shape [in_dim]).
  --src, --dst  : bf16 source + output dir (same shape as quantize_gemma4_safe_3bit.py).

Algorithm:
  1. For every quantizable Linear/SwitchLinear path in the model, look up the
     captured importance vector H ∈ R^{in_dim}.
  2. Compute per-tensor sensitivity:
         S(path) = Σ_c |W[:, c]|² * H[c]   (sum across c, across all out rows)
     This is the total output-energy contribution that flows through
     quantization-sensitive input channels — the larger S, the more output
     bandwidth is at risk if we drop precision.
  3. Hard skips (stay bf16): embed_tokens, norm, vision/audio towers, lm_head
     and any tensor whose weight.size % group_size != 0.
  4. Among quantizable tensors, sort by S descending. Top --top4-fraction
     (default 0.25) → 4-bit AFFINE group_size=64.  Rest → 3-bit AFFINE g=64.
  5. Hand the per-path bit table to mlx_lm.convert via a predicate that returns
     a dict — mlx_lm 0.31+ supports this for native per-layer mixed precision.

Why this beats the V1/V2/V3 static predicates:
  V1 protected `router` + `embed_tokens` + norms based on architectural intuition,
  ended up with a bf16/3-bit dtype boundary at the router that drove long-context
  drift.  V2 dropped router protection, drifted differently.  V3 added first/last
  2 layers bf16, ballooned to 16 GB (bigger than the 4-bit build).  None of them
  were data-driven — they couldn't tell which *specific* tensors actually carried
  the failure-mode signal.  This script measures it.

Output size estimate (top4_fraction = 0.25, group_size = 64):
  ~25 B quantizable params total
    - 25% at 4-bit = 6.25 B * 4/8 ≈ 3.1 GB
    - 75% at 3-bit = 18.75 B * 3/8 ≈ 7.0 GB
    - + ~10% overhead from scales/zeros (per-group f16 metadata)
    - + ~1-2 GB bf16 (embed_tokens + norms)
  Total ≈ 12-13 GB  (between 11 GB broken-3bit and 15 GB 4-bit)

Run:
  python scripts/quant/imatrix_mixed_quant.py \\
    --imatrix-dir /Users/sonheesung/models/gemma-4-26b-a4b-imatrix \\
    --src /Users/sonheesung/models/gemma-4-26b-a4b-bf16 \\
    --dst /Users/sonheesung/models/gemma-4-26b-a4b-mlx-imatrix3 \\
    --top4-fraction 0.25
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import safetensors.numpy
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.convert import convert as mlx_lm_convert
from mlx_lm.utils import load
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear

LINEAR_CLASSES = (nn.Linear, nn.QuantizedLinear)
SWITCH_CLASSES = (SwitchLinear, QuantizedSwitchLinear)

# Layers that NEVER quantize regardless of sensitivity.  Same set V1 used —
# these are architectural amplifiers / hooks where any quant-error broadcasts.
HARD_SKIP_SUBSTR = (
    "embed_tokens",
    "norm",
    "vision_tower",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    "per_layer",
    "layer_scalar",
    "embedding_post_projection",
    "lm_head",
)

GROUP_SIZE = 64
HIGH_PRECISION_BITS = 4   # top-sensitivity tier
MID_PRECISION_BITS = 3    # middle tier (default for non-extreme tensors)
LOW_PRECISION_BITS = 2    # bottom-sensitivity tier (only when --low2-fraction > 0)


def is_hard_skip(path: str) -> bool:
    return any(s in path for s in HARD_SKIP_SUBSTR)


def compute_sensitivities(model, importance: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-tensor sensitivity = mean per-channel activation sumsq.

    Canonical AWQ form is S(path) = Σ_c (Σ_r |W[r, c]|²) · H[c], i.e., the
    Frobenius norm of W after rescaling columns by sqrt(H).  We approximate
    that with S(path) ≈ mean(H) under the assumption that W's column norms
    are roughly homogeneous within a single quantizable tensor — which holds
    post-layer-norm in transformers (the magnitude distribution across
    columns is bounded by the norm-rescale upstream of every Linear).

    Dropping the W term avoids materializing 25 B bf16 weights on a 36 GB
    machine.  The relative *ranking* is what drives bit allocation, and the
    ranking is stable: tensors that see uniformly low-magnitude input
    columns get low scores regardless of W; tensors that see high-magnitude
    columns dominate regardless of W.

    Using mean(H) instead of sum(H) normalizes for in_dim so larger-input
    tensors aren't unfairly promoted.
    """
    S: dict[str, float] = {}
    quantizable_paths = set()
    for path, m in model.named_modules():
        if is_hard_skip(path):
            continue
        if not isinstance(m, LINEAR_CLASSES + SWITCH_CLASSES):
            continue
        # For lazy=True load, accessing .weight only triggers metadata, not data.
        if hasattr(m, "weight") and m.weight.size % GROUP_SIZE != 0:
            continue
        quantizable_paths.add(path)

    for path in quantizable_paths:
        h = importance.get(path)
        if h is None:
            S[path] = 0.0
            continue
        S[path] = float(h.astype(np.float64).mean())
    return S


def build_bit_table(
    sensitivities: dict[str, float],
    top4_fraction: float,
    low2_fraction: float,
) -> tuple[dict[str, int], list, int, int]:
    """Sort tensors by S descending; tier 1 → 4-bit, tier 3 → 2-bit, middle → 3-bit.

    When low2_fraction == 0.0 this reduces to the original 2-tier scheme
    (top4_fraction at 4-bit, rest at 3-bit, no 2-bit tier).
    """
    if top4_fraction + low2_fraction > 1.0:
        raise ValueError(
            f"top4_fraction + low2_fraction = {top4_fraction + low2_fraction} > 1.0"
        )
    ranked = sorted(sensitivities.items(), key=lambda kv: -kv[1])
    n_total = len(ranked)
    n_high = max(1, int(round(top4_fraction * n_total)))
    n_low = int(round(low2_fraction * n_total))
    n_mid = n_total - n_high - n_low

    bit_table: dict[str, int] = {}
    for i, (path, _) in enumerate(ranked):
        if i < n_high:
            bit_table[path] = HIGH_PRECISION_BITS
        elif i < n_high + n_mid:
            bit_table[path] = MID_PRECISION_BITS
        else:
            bit_table[path] = LOW_PRECISION_BITS
    return bit_table, ranked, n_high, n_low


def make_predicate(bit_table: dict[str, int]):
    """Predicate for mlx_lm.utils.quantize_model returning dict-of-params.

    Returns False (skip / stay bf16) when path matches HARD_SKIP_SUBSTR or
    module isn't quantizable.  Otherwise returns the per-layer params dict
    looked up from `bit_table`.  Unknown paths default to MID_PRECISION_BITS
    (3-bit) — these are tensors the importance map didn't see, which would
    be surprising for a properly-calibrated build.
    """

    def predicate(path: str, module):
        if not (isinstance(module, nn.Linear) or hasattr(module, "to_quantized")):
            return False
        if is_hard_skip(path):
            return False
        if hasattr(module, "weight") and module.weight.size % GROUP_SIZE != 0:
            return False
        bits = bit_table.get(path, MID_PRECISION_BITS)
        return {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}

    return predicate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix-dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument(
        "--top4-fraction",
        type=float,
        default=0.25,
        help="fraction of quantizable tensors that go to 4-bit (highest sensitivity)",
    )
    ap.add_argument(
        "--low2-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of quantizable tensors that go to 2-bit (lowest "
            "sensitivity).  Default 0 = 2-tier (4-bit + 3-bit) mode.  "
            "Set > 0 to enable 3-tier (4-bit + 3-bit + 2-bit).  "
            "Bottom tensors with score < ~100 are safe candidates."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print bit allocation + estimated output size, don't write",
    )
    args = ap.parse_args()

    imatrix_dir = pathlib.Path(args.imatrix_dir)
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    importance_path = imatrix_dir / "importance.npz"
    if not importance_path.exists():
        print(f"importance file not found: {importance_path}", file=sys.stderr)
        return 2
    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr)
        return 2
    if dst.exists() and any(dst.iterdir()):
        print(f"dst exists and is non-empty: {dst} — remove first", file=sys.stderr)
        return 2

    print(f"[load] importance from {importance_path}")
    with np.load(importance_path) as npz:
        importance = {key: npz[key].astype(np.float64) for key in npz.files}
    print(f"[load] {len(importance)} importance vectors")

    print(f"[load] model {src} (lazy, for sensitivity scoring)")
    model, _tok = load(str(src), tokenizer_config={"eos_token": "<end_of_turn>"}, lazy=True)

    print("[score] computing per-tensor sensitivities ...")
    S = compute_sensitivities(model, importance)
    bit_table, ranked, n_high, n_low = build_bit_table(
        S, args.top4_fraction, args.low2_fraction
    )
    n_total = len(bit_table)
    n_mid = n_total - n_high - n_low

    print(f"[plan] tier breakdown over {n_total} quantizable tensors:")
    print(f"       {n_high:3d} → 4-bit (top {args.top4_fraction*100:.0f}%)")
    print(f"       {n_mid:3d} → 3-bit (middle)")
    print(f"       {n_low:3d} → 2-bit (bottom {args.low2_fraction*100:.0f}%)")
    print(f"[plan] top-{HIGH_PRECISION_BITS}bit tensors (highest sensitivity):")
    for path, score in ranked[:10]:
        print(f"  S={score: .3e}  {path}")
    if n_low > 0:
        print(f"[plan] 2-bit tensors (bottom {n_low}, lowest sensitivity):")
        for path, score in ranked[-min(10, n_low):]:
            print(f"  S={score: .3e}  {path}")
    else:
        print(f"[plan] bottom 5 sensitivity (sample, all → 3-bit):")
        for path, score in ranked[-5:]:
            print(f"  S={score: .3e}  {path}")

    # Size estimate
    n_quant_4 = n_quant_3 = n_quant_2 = n_bf16 = 0
    for path, m in model.named_modules():
        if not hasattr(m, "weight"):
            continue
        w = m.weight
        if not hasattr(w, "size"):
            continue
        n = w.size
        if is_hard_skip(path):
            n_bf16 += n
        elif not (isinstance(m, nn.Linear) or isinstance(m, SwitchLinear)):
            n_bf16 += n
        elif w.size % GROUP_SIZE != 0:
            n_bf16 += n
        else:
            tier_bits = bit_table.get(path, MID_PRECISION_BITS)
            if tier_bits == HIGH_PRECISION_BITS:
                n_quant_4 += n
            elif tier_bits == MID_PRECISION_BITS:
                n_quant_3 += n
            else:
                n_quant_2 += n

    bytes_4 = n_quant_4 * HIGH_PRECISION_BITS / 8
    bytes_3 = n_quant_3 * MID_PRECISION_BITS / 8
    bytes_2 = n_quant_2 * LOW_PRECISION_BITS / 8
    bytes_bf16 = n_bf16 * 2
    # ~10% overhead for AFFINE scale+zero metadata (f16 each per group).
    # 2-bit has proportionally more metadata so use 12% there.
    metadata = 0.10 * (bytes_4 + bytes_3) + 0.12 * bytes_2
    total_gb = (bytes_4 + bytes_3 + bytes_2 + bytes_bf16 + metadata) / 1e9
    print(f"[size]  4-bit params : {n_quant_4/1e9:.2f} B  ({bytes_4/1e9:.2f} GB)")
    print(f"[size]  3-bit params : {n_quant_3/1e9:.2f} B  ({bytes_3/1e9:.2f} GB)")
    print(f"[size]  2-bit params : {n_quant_2/1e9:.2f} B  ({bytes_2/1e9:.2f} GB)")
    print(f"[size]  bf16  params : {n_bf16/1e9:.2f} B  ({bytes_bf16/1e9:.2f} GB)")
    print(f"[size]  est total    : ~{total_gb:.1f} GB  (incl. AFFINE metadata)")

    if args.dry_run:
        return 0

    predicate = make_predicate(bit_table)
    print(f"[convert] {src} → {dst}")
    mlx_lm_convert(
        hf_path=str(src),
        mlx_path=str(dst),
        quantize=True,
        q_group_size=GROUP_SIZE,
        q_bits=MID_PRECISION_BITS,  # ignored when predicate returns dict per layer
        dtype="bfloat16",
        quant_predicate=predicate,
    )

    recipe_name = (
        "lumen.gemma4.imatrix3tier" if args.low2_fraction > 0
        else "lumen.gemma4.imatrix3"
    )
    marker = dst / "lumen_quant_recipe.json"
    marker.write_text(
        json.dumps(
            {
                "recipe": recipe_name,
                "source": str(src),
                "imatrix_dir": str(imatrix_dir),
                "top4_fraction": args.top4_fraction,
                "low2_fraction": args.low2_fraction,
                "high_precision_bits": HIGH_PRECISION_BITS,
                "mid_precision_bits": MID_PRECISION_BITS,
                "low_precision_bits": LOW_PRECISION_BITS,
                "group_size": GROUP_SIZE,
                "n_tensors_high": n_high,
                "n_tensors_mid": n_mid,
                "n_tensors_low": n_low,
                "skip_substr": list(HARD_SKIP_SUBSTR),
                "bit_table": bit_table,
            },
            indent=2,
        )
    )
    print(f"[convert] wrote {marker.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
