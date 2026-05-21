"""Filter awq_scales.npz by group-name regex or member-bits — ablation tool.

Takes the full awq_scales.npz + .json from awq_search.py and emits a subset
where only groups matching the filter are kept. Used for ablating which
group-type / depth / bit-tier combination of AWQ actually improves PPL.

Examples:
    # Keep only qkv_input groups
    python scripts/quant/awq_filter_scales.py \\
      --src awq_scales.npz --pattern '\\.qkv_input$' --out awq_scales_qkv.npz

    # Keep only mlp_down groups
    --pattern '\\.mlp_down$'

    # Keep only layers 0-9 (early layers)
    --pattern '\\.layers\\.[0-9]\\.'

    # Keep only groups whose JOINT MSE improvement exceeded 15%
    --min-improvement-pct 15.0

    # Keep only groups where ALL members are 3-bit
    --bits 3

Output structure mirrors awq_search.py's: .npz + sidecar .json
(report['groups'] retains all entries but `kept=false` for filtered-out ones).
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source awq_scales.npz from awq_search.py")
    ap.add_argument("--out", required=True, help="output .npz path (sibling .json auto-written)")
    ap.add_argument("--pattern", type=str, default=None,
                    help="regex matched against group name (e.g. '\\.qkv_input$')")
    ap.add_argument("--invert", action="store_true",
                    help="invert: keep groups NOT matching --pattern")
    ap.add_argument("--min-improvement-pct", type=float, default=None,
                    help="keep only groups with proxy improvement_pct >= this")
    ap.add_argument("--bits", type=int, default=None,
                    help="keep only groups where ALL members have this bit count")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    src_report = src.with_suffix(".json")
    if not src.exists() or not src_report.exists():
        print(f"missing src .npz or .json sidecar: {src}", file=sys.stderr)
        return 2

    print(f"[load] {src}")
    with np.load(src) as npz:
        scales = {k: npz[k].astype(np.float32) for k in npz.files if k != "_empty"}
    report = json.loads(src_report.read_text())
    groups = report["groups"]
    print(f"[load] {len(scales)} kept scales, {len(groups)} reported groups")

    pat = re.compile(args.pattern) if args.pattern else None

    new_scales: dict[str, np.ndarray] = {}
    kept_names: list[str] = []
    dropped_names: list[str] = []
    for name, scale_arr in scales.items():
        g_meta = groups.get(name, {})

        keep = True
        if pat is not None:
            matched = pat.search(name) is not None
            if args.invert:
                keep = keep and (not matched)
            else:
                keep = keep and matched
        if args.min_improvement_pct is not None:
            keep = keep and (g_meta.get("improvement_pct", 0.0) >= args.min_improvement_pct)
        if args.bits is not None:
            members = g_meta.get("members", [])
            if not members or not all(m.get("bits") == args.bits for m in members):
                keep = False

        if keep:
            new_scales[name] = scale_arr
            kept_names.append(name)
        else:
            dropped_names.append(name)

    print(f"[filter] kept {len(kept_names)} / dropped {len(dropped_names)}")
    if kept_names:
        print(f"  first kept: {kept_names[:5]}")
    if dropped_names:
        print(f"  first dropped: {dropped_names[:5]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if new_scales:
        np.savez_compressed(out, **new_scales)
    else:
        np.savez_compressed(out, _empty=np.zeros(0, dtype=np.float32))
        print("[warn] no groups matched — writing sentinel", file=sys.stderr)

    # Sidecar JSON: same structure as awq_search.py output, but flag dropped
    # groups as kept=false so imatrix_mixed_quant skips them at apply time.
    new_report = dict(report)
    new_groups = {}
    for name, meta in groups.items():
        meta2 = dict(meta)
        meta2["kept"] = name in new_scales
        new_groups[name] = meta2
    new_report["groups"] = new_groups
    new_report["filter"] = {
        "src": str(src),
        "pattern": args.pattern,
        "invert": args.invert,
        "min_improvement_pct": args.min_improvement_pct,
        "bits": args.bits,
        "n_kept": len(kept_names),
        "n_dropped": len(dropped_names),
    }
    out.with_suffix(".json").write_text(json.dumps(new_report, indent=2))
    print(f"[done] wrote {out.name} + {out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
