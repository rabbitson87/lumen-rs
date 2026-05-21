"""Apply AWQ group scales: scale member weights, absorb 1/s into adjacent ops.

Math (per kept group):

    s ∈ R^{in_dim}                                    (from awq_search output)

    For each Linear member with weight W [out, in]:
        W'  = W · diag(s)                             (broadcast s along axis=1 / in)
    For each SwitchLinear member with weight W [E, out, in]:
        W'  = W · diag(s)                             (broadcast s along axis=2 / in)

    Absorption (kind):
      "norm_weight":      RMSNorm γ ← γ / s            (elementwise, len = in_dim)
      "linear_out_axis":  Linear  W ← W / s[:, None]   (divide axis=0 / out by s)

The math is an exact identity in bf16 forward (up to rounding) — verify with
`tests/awq_apply_fp_parity` before running long quantize.

This script outputs a bf16 model with AWQ transforms applied; downstream
quantization (imatrix_mixed_quant.py) sees a "regular" model with shifted
magnitude distribution where salient input channels are amplified, producing
lower quantization error at the same bit budget.

Run:
    python scripts/quant/awq_apply.py \\
      --src         /Users/sonheesung/models/gemma-4-26b-a4b-bf16 \\
      --scales      /Users/sonheesung/models/gemma-4-26b-a4b-imatrix/awq_scales.npz \\
      --out-dir     /Users/sonheesung/models/gemma-4-26b-a4b-bf16-awq
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn  # noqa: F401
from mlx_lm.utils import load, save_model
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from imatrix_mixed_quant import LINEAR_CLASSES, SWITCH_CLASSES  # noqa: E402


# Files copied through verbatim from src → out_dir.  Anything we don't recognize
# is also copied (best-effort).  model.safetensors* are SKIPPED — save_model
# writes fresh shards from the AWQ-transformed parameters.
COPY_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "generation_config.json",
    "chat_template.json",
    "preprocessor_config.json",
)
SKIP_PATTERNS = (
    "model.safetensors",
    "model-",
    "model.safetensors.index.json",
)


def _is_safetensor_artifact(name: str) -> bool:
    return any(name.startswith(p) or p in name for p in SKIP_PATTERNS) and name.endswith((
        ".safetensors", ".safetensors.index.json"
    ))


def _copy_metadata(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Copy non-weight artifacts from src to dst.  Returns count copied."""
    n = 0
    for p in src.iterdir():
        if not p.is_file():
            continue
        if _is_safetensor_artifact(p.name):
            continue
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def _apply_member_scale(module, s: mx.array) -> None:
    """Multiply module.weight by s on the innermost (input) axis.

    Linear:        weight shape [out, in]       → w * s[None, :]
    SwitchLinear:  weight shape [E, out, in]    → w * s[None, None, :]
    """
    w = module.weight
    orig_dtype = w.dtype
    if isinstance(module, SWITCH_CLASSES):
        w_new = (w.astype(mx.float32) * s[None, None, :]).astype(orig_dtype)
    else:
        w_new = (w.astype(mx.float32) * s[None, :]).astype(orig_dtype)
    module.weight = w_new


def _apply_absorb(module, s: mx.array, kind: str) -> None:
    """Divide module.weight by s according to absorption kind.

    "norm_weight":     γ shape [in_dim] elementwise   γ ← γ / s
    "linear_out_axis": W shape [out, in], divide rows  W ← W / s[:, None]
                       (len(s) must equal out)
    """
    w = module.weight
    orig_dtype = w.dtype
    if kind == "norm_weight":
        if w.ndim != 1 or w.shape[0] != s.shape[0]:
            raise ValueError(
                f"norm_weight absorption: expected γ shape [in_dim]={s.shape[0]}, "
                f"got {w.shape}"
            )
        w_new = (w.astype(mx.float32) / s).astype(orig_dtype)
    elif kind == "linear_out_axis":
        if w.ndim != 2 or w.shape[0] != s.shape[0]:
            raise ValueError(
                f"linear_out_axis absorption: expected weight shape [out, in] with "
                f"out={s.shape[0]}, got {w.shape}"
            )
        w_new = (w.astype(mx.float32) / s[:, None]).astype(orig_dtype)
    else:
        raise ValueError(f"unknown absorb kind: {kind!r}")
    module.weight = w_new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="bf16 source model path")
    ap.add_argument(
        "--scales",
        required=True,
        help="awq_scales.npz from awq_search.py (sibling .json with absorption metadata)",
    )
    ap.add_argument("--out-dir", required=True, help="output dir for AWQ-applied bf16 model")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve groups + validate shapes but don't apply or save",
    )
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    scales_path = pathlib.Path(args.scales)
    out_dir = pathlib.Path(args.out_dir)
    report_path = scales_path.with_suffix(".json")

    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr)
        return 2
    if not scales_path.exists():
        print(f"scales not found: {scales_path}", file=sys.stderr)
        return 2
    if not report_path.exists():
        print(f"report sidecar not found: {report_path}", file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()) and not args.dry_run:
        print(f"out_dir exists and is non-empty: {out_dir} — remove first", file=sys.stderr)
        return 2

    print(f"[load] scales from {scales_path}")
    with np.load(scales_path) as npz:
        scales_np = {k: npz[k].astype(np.float32) for k in npz.files if k != "_empty"}
    report = json.loads(report_path.read_text())
    groups_meta: dict = report["groups"]
    print(f"[load] {len(scales_np)} kept group scales, {len(groups_meta)} total reported")

    if not scales_np:
        print("[noop] no AWQ scales to apply — copying src → out_dir verbatim",
              file=sys.stderr)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in src.iterdir():
                if p.is_file():
                    shutil.copy2(p, out_dir / p.name)
            print(f"[done] copied src → {out_dir}")
        return 0

    print(f"[load] model {src} (lazy)")
    model, _tok = load(
        str(src),
        tokenizer_config={"eos_token": "<end_of_turn>"},
        lazy=True,
    )

    path_to_module: dict = dict(model.named_modules())
    print(f"[map] {len(path_to_module)} module paths")

    n_applied = 0
    n_failed = 0
    n_members_scaled = 0
    t0 = time.time()

    for group_name, scale_arr in scales_np.items():
        g_meta = groups_meta.get(group_name)
        if g_meta is None or not g_meta.get("kept", False):
            print(f"  [skip] {group_name}: not in kept set", file=sys.stderr)
            continue

        s = mx.array(scale_arr.astype(np.float32))
        absorb = g_meta["absorb"]
        target_path = absorb["target_path"]
        kind = absorb["kind"]

        target_mod = path_to_module.get(target_path)
        if target_mod is None:
            print(f"  [fail] {group_name}: absorb target {target_path} not in model",
                  file=sys.stderr)
            n_failed += 1
            continue

        # Scale members first (before absorbing — order is mathematically
        # irrelevant, but doing members first means any failure leaves the
        # γ/up_proj absorber untouched and forward stays correct).
        member_modules = []
        member_resolved_ok = True
        for m_info in g_meta["members"]:
            mod = path_to_module.get(m_info["path"])
            if mod is None:
                print(f"  [fail] {group_name}: member {m_info['path']} not in model",
                      file=sys.stderr)
                member_resolved_ok = False
                break
            in_dim = mod.weight.shape[-1]
            if in_dim != s.shape[0]:
                print(f"  [fail] {group_name}: in_dim mismatch on {m_info['path']} "
                      f"(W={in_dim}, s={s.shape[0]})", file=sys.stderr)
                member_resolved_ok = False
                break
            member_modules.append(mod)
        if not member_resolved_ok:
            n_failed += 1
            continue

        if args.dry_run:
            print(f"  [dry] {group_name}: would scale {len(member_modules)} members, "
                  f"absorb into {target_path} ({kind})")
            n_applied += 1
            n_members_scaled += len(member_modules)
            continue

        try:
            for mod in member_modules:
                _apply_member_scale(mod, s)
                n_members_scaled += 1
            _apply_absorb(target_mod, s, kind)
        except Exception as e:
            print(f"  [fail] {group_name}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        n_applied += 1
        if n_applied % 25 == 0:
            print(f"  [{n_applied}/{len(scales_np)}] applied; "
                  f"elapsed={time.time()-t0:.1f}s  last={group_name}")

    print(f"[done] applied {n_applied} groups ({n_members_scaled} member weights scaled), "
          f"{n_failed} failed in {time.time()-t0:.1f}s")

    if args.dry_run:
        return 0 if n_failed == 0 else 1

    print("[eval] forcing materialization of mutated parameters ...")
    mx.eval(model.parameters())

    print(f"[save] writing AWQ-applied bf16 to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(out_dir, model, donate_model=True)

    n_copied = _copy_metadata(src, out_dir)
    print(f"[save] copied {n_copied} metadata files")

    # Stamp the output with an AWQ marker so downstream tools can detect
    # AWQ provenance / refuse double-application.
    marker = out_dir / "lumen_awq_apply.json"
    marker.write_text(json.dumps({
        "src": str(src),
        "scales": str(scales_path),
        "n_groups_applied": n_applied,
        "n_member_weights_scaled": n_members_scaled,
        "n_groups_failed": n_failed,
        "absorption_summary": {
            "norm_weight": sum(
                1 for g in scales_np
                if groups_meta.get(g, {}).get("absorb", {}).get("kind") == "norm_weight"
            ),
            "linear_out_axis": sum(
                1 for g in scales_np
                if groups_meta.get(g, {}).get("absorb", {}).get("kind") == "linear_out_axis"
            ),
        },
    }, indent=2))
    print(f"[done] wrote {marker.name}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
