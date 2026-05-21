"""Mixed 3-bit / 4-bit quantization for Gemma 4 26B-A4B driven by imatrix stats.

Inputs:
  --imatrix-dir : directory produced by imatrix_capture_gemma4.py containing
                  importance.npz (per-path activation sumsq, shape [in_dim]).
  --src, --dst  : bf16 source + output dir (same shape as quantize_gemma4_safe_3bit.py).
  --apply-awq-scales (optional)
                : path to awq_scales.npz from awq_search.py.  When given,
                  AWQ per-input-channel scales are applied to model weights
                  in-memory before quantization — avoids materializing a
                  48 GB AWQ-only bf16 intermediate directory.

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
import os
import pathlib
import shutil
import sys

import numpy as np
import safetensors.numpy
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.convert import convert as mlx_lm_convert
from mlx_lm.utils import load, load_config, quantize_model, save_model, save_config
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear

LINEAR_CLASSES = (nn.Linear, nn.QuantizedLinear)
SWITCH_CLASSES = (SwitchLinear, QuantizedSwitchLinear)

# Layers that NEVER quantize regardless of sensitivity.  Same set V1 used —
# these are architectural amplifiers / hooks where any quant-error broadcasts.
HARD_SKIP_SUBSTR = (
    # embed_tokens intentionally NOT skipped: Gemma 4 ties lm_head to it, so
    # leaving embed_tokens at bf16 makes tied lm_head matmul read 1.07 GB/step
    # (262K vocab × 2048 hidden × 2 byte), ~1.7 ms BW cost per decode step.
    # Quantizing it brings parity with mlx-community uniform 4-bit speed.
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

# Per-tensor bit override for tensors the sensitivity ranking misses.
# embed_tokens is the canonical case: its imatrix sumsq is low (each token
# only activates one row), so the data-driven ranking would push it to
# 3-bit, but tied lm_head reads every row on every step. Empirical PPL
# vs baseline imatrix3plus (mean across seeds 7/42/123) on Gemma 4 26B-A4B:
#   bf16 embed: −17.33 ppl   (slow: tied lm_head = 1.07 GB read/step)
#   8-bit embed: TBD          (middle: ~540 MB read/step)
#   4-bit embed: −12.51 ppl   (fast: ~270 MB read/step)
#   3-bit embed: −3.15 ppl    (collapsed; do not use)
# Tunable via the dict below — substring match against the path.
FORCE_BITS_SUBSTR: dict[str, int] = {
    "embed_tokens": int(os.environ.get("LUMEN_EMBED_BITS", "4")),
}

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

    forced_overrides: list[tuple[str, int]] = []
    for path in bit_table:
        for sub, forced_bits in FORCE_BITS_SUBSTR.items():
            if sub in path and bit_table[path] != forced_bits:
                bit_table[path] = forced_bits
                forced_overrides.append((path, forced_bits))
                break
    if forced_overrides:
        print(f"[plan] forced bit override on {len(forced_overrides)} tensor(s):")
        for path, b in forced_overrides:
            print(f"  → {path}  ({b}-bit)")

    n_high_final = sum(1 for b in bit_table.values() if b == HIGH_PRECISION_BITS)
    return bit_table, ranked, n_high_final, n_low


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
        # FORCE_BITS_SUBSTR takes priority over bit_table lookup.
        # Sensitivity ranking only considers Linear/SwitchLinear, so non-Linear
        # quantizable layers (e.g. nn.Embedding) aren't in bit_table at all
        # and would fall through to MID_PRECISION_BITS unless forced here.
        for sub, forced_bits in FORCE_BITS_SUBSTR.items():
            if sub in path:
                return {"group_size": GROUP_SIZE, "bits": forced_bits, "mode": "affine"}
        bits = bit_table.get(path, MID_PRECISION_BITS)
        return {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}

    return predicate


def _apply_awq_scales_inplace(model, scales_path: pathlib.Path) -> tuple[int, int]:
    """Apply AWQ per-input-channel scales to model weights in-memory.

    Reads awq_scales.npz + sibling .json (from awq_search.py).  For each kept
    group: multiplies member weights by s along innermost axis, then divides
    the absorption target weight (RMSNorm γ elementwise OR Linear out-axis).

    Returns (n_groups_applied, n_groups_failed).  Math identity verified by
    scripts/quant/test_awq_apply_parity.py.
    """
    report_path = scales_path.with_suffix(".json")
    if not scales_path.exists():
        raise FileNotFoundError(f"AWQ scales not found: {scales_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"AWQ report sidecar not found: {report_path}")

    with np.load(scales_path) as npz:
        scales_np = {k: npz[k].astype(np.float32) for k in npz.files if k != "_empty"}
    groups_meta: dict = json.loads(report_path.read_text())["groups"]
    if not scales_np:
        print("[awq] no scales to apply (sentinel .npz)", file=sys.stderr)
        return 0, 0

    path_to_module = dict(model.named_modules())
    n_applied = 0
    n_failed = 0

    for group_name, scale_arr in scales_np.items():
        g_meta = groups_meta.get(group_name)
        if not g_meta or not g_meta.get("kept", False):
            continue

        s = mx.array(scale_arr.astype(np.float32))
        absorb = g_meta["absorb"]
        target_mod = path_to_module.get(absorb["target_path"])
        if target_mod is None:
            print(f"  [awq fail] {group_name}: absorb target {absorb['target_path']} "
                  f"not in model", file=sys.stderr)
            n_failed += 1
            continue

        member_mods = []
        ok = True
        for m_info in g_meta["members"]:
            mod = path_to_module.get(m_info["path"])
            if mod is None or mod.weight.shape[-1] != s.shape[0]:
                print(f"  [awq fail] {group_name}: member {m_info['path']} "
                      f"missing or shape mismatch", file=sys.stderr)
                ok = False
                break
            member_mods.append(mod)
        if not ok:
            n_failed += 1
            continue

        try:
            # Scale members: W ← W · diag(s)  along innermost (input) axis.
            for mod in member_mods:
                w = mod.weight
                if isinstance(mod, SWITCH_CLASSES):
                    mod.weight = (w.astype(mx.float32) * s[None, None, :]).astype(w.dtype)
                else:
                    mod.weight = (w.astype(mx.float32) * s[None, :]).astype(w.dtype)

            # Absorb 1/s into the upstream op.
            old = target_mod.weight
            kind = absorb["kind"]
            if kind == "norm_weight":
                target_mod.weight = (old.astype(mx.float32) / s).astype(old.dtype)
            elif kind == "linear_out_axis":
                target_mod.weight = (old.astype(mx.float32) / s[:, None]).astype(old.dtype)
            else:
                raise ValueError(f"unknown absorb kind: {kind!r}")
        except Exception as e:
            print(f"  [awq fail] {group_name}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        n_applied += 1

    print(f"[awq] applied {n_applied} group scales, {n_failed} failed "
          f"(out of {len(scales_np)} kept in scales file)")
    return n_applied, n_failed


def _copy_tokenizer_files(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Copy non-weight metadata (tokenizer, generation_config, …) from src to dst.

    Skips model.safetensors* (save_model writes fresh shards) and config.json
    (save_config writes the quant-augmented version).
    """
    n = 0
    for p in src.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name == "config.json":
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(p, dst / name)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix-dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument(
        "--apply-awq-scales",
        type=str,
        default=None,
        help="optional path to awq_scales.npz (sibling .json auto-resolved). "
             "When given, AWQ scales are applied in-memory before quantize, "
             "skipping the 48 GB AWQ-only bf16 intermediate.",
    )
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

    awq_applied = 0
    awq_failed = 0
    awq_scales_path: pathlib.Path | None = None

    if args.apply_awq_scales:
        # In-memory path: load full bf16 → apply AWQ → quantize_model → save manually.
        # Avoids producing a 48 GB AWQ-only intermediate dir.
        awq_scales_path = pathlib.Path(args.apply_awq_scales)
        print(f"[awq] loading FULL bf16 model {src} (not lazy — needed for in-place scale apply)")
        full_model, _tok = load(
            str(src), tokenizer_config={"eos_token": "<end_of_turn>"}, lazy=False
        )
        awq_applied, awq_failed = _apply_awq_scales_inplace(full_model, awq_scales_path)
        mx.eval(full_model.parameters())

        print(f"[convert] in-memory quantize → {dst}")
        config = load_config(src)
        full_model, quantized_config = quantize_model(
            full_model,
            config,
            group_size=GROUP_SIZE,
            bits=MID_PRECISION_BITS,
            mode="affine",
            quant_predicate=predicate,
        )

        dst.mkdir(parents=True, exist_ok=True)
        save_model(dst, full_model, donate_model=True)
        save_config(quantized_config, dst / "config.json")
        n_copied = _copy_tokenizer_files(src, dst)
        print(f"[convert] saved weights + config; copied {n_copied} tokenizer/metadata files")
    else:
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

    base_recipe = (
        "lumen.gemma4.imatrix3tier" if args.low2_fraction > 0
        else "lumen.gemma4.imatrix3"
    )
    recipe_name = f"{base_recipe}-awq" if args.apply_awq_scales else base_recipe
    marker = dst / "lumen_quant_recipe.json"
    marker_payload = {
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
    }
    if args.apply_awq_scales:
        marker_payload["awq"] = {
            "scales_path": str(awq_scales_path),
            "n_groups_applied": awq_applied,
            "n_groups_failed": awq_failed,
        }
    marker.write_text(json.dumps(marker_payload, indent=2))
    print(f"[convert] wrote {marker.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
