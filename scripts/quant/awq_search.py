"""Group-aware AWQ scale search on top of imatrix calibration.

In Gemma 4 multiple Linear consumers share the same input source (q/k/v_proj
all read input_layernorm output; gate/up_proj all read pre_feedforward_layernorm
output).  AWQ needs ONE per-input-channel scale per shared-input group so the
absorption (γ_norm /= s) is internally consistent.  This script therefore
searches α PER GROUP, with the objective summed across all group members.

Groups (Gemma 4 26B-A4B, Phase A — Dense only):

  per decoder layer i ∈ [0, num_hidden_layers):
    G1) qkv_input    members = {q_proj, k_proj, v_proj}      shared input from input_layernorm
    G2) mlp_input    members = {mlp.gate_proj, mlp.up_proj}  shared input from pre_feedforward_layernorm
    G3) mlp_down     members = {mlp.down_proj}               input = geglu(gate, up); absorb into up_proj OUT-axis

  Skipped in Phase A:
    - o_proj          (post-SDPA input; v_norm RMS interaction makes per-channel fold non-trivial)
    - router.proj     (small, already 8-bit per Model.quant_predicate)
    - experts.*       (MoE expert SwitchLinear — Phase B with --include-experts)

Algorithm (per group):
    s_c(α) = (mean_abs_c)^α     (mean_abs shared across members — they share input)
    normalize  s ← s / sqrt(max(s)·min(s))      geometric-mean = 1
    clip       s ← clamp(s, 1e-3, 1e3)

    For each member with target bits b:
        W_rec(b, s) = dequant(quant(W·diag(s), b)) · diag(1/s)
        loss_m      = Σ_{r,c} (W_rec − W)²[r,c] · mean_abs[c]    activation-weighted col-MSE

    group_loss(α) = Σ_m loss_m
    α* = argmin group_loss(α)
    keep s when group_loss(α*) < group_loss(α=0) · (1 − min_improvement_pct/100)

Outputs:
    <out>             .npz  — per group  "{group_name}::s" → [in_dim] f32
    <out>.json sidecar     — group α / per-member loss / improvement / absorption target

Run:
    python scripts/quant/awq_search.py \\
      --imatrix-dir /Users/sonheesung/models/gemma-4-26b-a4b-imatrix \\
      --src         /Users/sonheesung/models/gemma-4-26b-a4b-bf16 \\
      --out         /Users/sonheesung/models/gemma-4-26b-a4b-imatrix/awq_scales.npz \\
      --top4-fraction 0.35
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from imatrix_mixed_quant import (  # noqa: E402
    LINEAR_CLASSES,
    SWITCH_CLASSES,
    GROUP_SIZE,
    HARD_SKIP_SUBSTR,  # noqa: F401
    is_hard_skip,
    compute_sensitivities,
    build_bit_table,
)


# ---------------------------------------------------------------------------
# Core ops
# ---------------------------------------------------------------------------

def _quantize_then_dequantize(w: mx.array, bits: int, group_size: int) -> mx.array:
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return mx.dequantize(q, scales, biases, group_size=group_size, bits=bits)


def _weighted_col_mse(W_orig: mx.array, W_rec: mx.array, w_col: mx.array) -> mx.array:
    err2 = (W_rec.astype(mx.float32) - W_orig.astype(mx.float32)) ** 2
    return (err2 * w_col[None, :]).sum()


def _compute_scale(mean_abs: mx.array, alpha: float) -> mx.array:
    """Per-AWQ paper: s_c = mean_abs_c^α, normalized to unit geometric mean, clipped.

    α = 0 → s = ones (no AWQ).  α = 1 → s = mean_abs (max amplification of salient cols).
    """
    if alpha == 0.0:
        return mx.ones_like(mean_abs)
    s = mx.power(mean_abs, alpha)
    s = s / mx.sqrt(s.max() * s.min())
    return mx.clip(s, 1e-3, 1e3)


def search_group(
    members: list[dict],
    scale_input: mx.array,
    loss_weight: mx.array,
    alpha_grid: list[float],
    group_size: int = 64,
) -> dict:
    """Joint α search across group members sharing one input.

    members:     list of dicts {'path', 'W', 'bits'}; W innermost dim must match.
    scale_input: per-input-channel statistic used in the AWQ scale formula
                 s_c = (scale_input_c)^α  (always E[|X_c|] = mean_abs by AWQ paper).
    loss_weight: per-input-channel statistic used in the proxy loss weighting.
                 'abs' moment → mean_abs;  'sq' moment → mean_sq.
                 Both shapes are [in_dim] and must match scale_input.shape.

    Returns dict with: best_s, best_alpha, baseline_loss_total, best_loss_total,
                       per_member_baseline, per_member_best, improvement_pct.
    """
    eps = 1e-5
    si = mx.clip(scale_input, eps, None).astype(mx.float32)
    lw = mx.clip(loss_weight, 0.0, None).astype(mx.float32)
    Wfs = [m["W"].astype(mx.float32) for m in members]
    paths = [m["path"] for m in members]
    bits_list = [m["bits"] for m in members]

    # Baseline (α = 0 → s = ones, no AWQ).
    baseline_losses: list[float] = []
    for Wf, bits in zip(Wfs, bits_list):
        W_rec = _quantize_then_dequantize(Wf, bits, group_size)
        loss = _weighted_col_mse(Wf, W_rec, lw)
        mx.eval(loss)
        baseline_losses.append(float(loss))
    baseline_total = sum(baseline_losses)

    best_alpha = 0.0
    best_total = baseline_total
    best_per_member = list(baseline_losses)
    best_s = mx.ones_like(si)

    for alpha in alpha_grid:
        if alpha == 0.0:
            continue
        s = _compute_scale(si, alpha)

        per_member: list[float] = []
        for Wf, bits in zip(Wfs, bits_list):
            W_scaled = Wf * s[None, :]
            W_hat = _quantize_then_dequantize(W_scaled, bits, group_size)
            W_rec = W_hat / s[None, :]
            loss = _weighted_col_mse(Wf, W_rec, lw)
            mx.eval(loss)
            per_member.append(float(loss))
        total = sum(per_member)
        if total < best_total:
            best_total = total
            best_alpha = alpha
            best_per_member = per_member
            best_s = s

    mx.eval(best_s)
    improvement_pct = 100.0 * (baseline_total - best_total) / max(baseline_total, 1e-12)
    return {
        "best_s": best_s,
        "best_alpha": best_alpha,
        "baseline_loss_total": baseline_total,
        "best_loss_total": best_total,
        "per_member_baseline": dict(zip(paths, baseline_losses)),
        "per_member_best": dict(zip(paths, best_per_member)),
        "improvement_pct": improvement_pct,
    }


# ---------------------------------------------------------------------------
# Group discovery — encodes Gemma 4 architecture (gemma4_text.DecoderLayer)
# ---------------------------------------------------------------------------

def _collect_group_members(
    layer: nn.Module,
    layer_prefix: str,
    attr_path: list[str],
    proj_names: list[str],
    bit_table: dict[str, int],
) -> list[dict]:
    """Resolve `attr_path` through `layer` and collect matching proj members.

    e.g., attr_path=["self_attn"], proj_names=["q_proj","k_proj","v_proj"]
          → returns members for q/k/v_proj on layer.self_attn
    Returns: list of {"path", "module", "bits"} (skips members not in bit_table).
    """
    cur: Any = layer
    for a in attr_path:
        cur = getattr(cur, a, None)
        if cur is None:
            return []

    out: list[dict] = []
    for proj in proj_names:
        sub = getattr(cur, proj, None)
        if sub is None:
            continue
        sub_path_inner = ".".join(attr_path + [proj])
        full_path = f"{layer_prefix}.{sub_path_inner}"
        if full_path not in bit_table:
            continue
        out.append({"path": full_path, "module": sub, "bits": bit_table[full_path]})
    return out


def _detect_layer_prefix(bit_table: dict[str, int]) -> str:
    """Detect the 'X.layers' prefix from any q_proj path in bit_table.

    Handles both layouts:
      - gemma4_text  →  "model.layers.{i}.self_attn.q_proj"        → "model.layers"
      - gemma4 (mm)  →  "language_model.model.layers.{i}.self_attn.q_proj"
                                                                  → "language_model.model.layers"
    """
    import re
    pat = re.compile(r"^(.+\.layers)\.\d+\.self_attn\.q_proj$")
    for p in bit_table:
        m = pat.match(p)
        if m:
            return m.group(1)
    raise RuntimeError(
        "could not detect '*.layers' prefix from bit_table — "
        "no q_proj path matched. Is this really a Gemma 4 model?"
    )


def _layer_indices(bit_table: dict[str, int], layer_prefix: str) -> list[int]:
    """Enumerate decoder layer indices appearing under the detected prefix."""
    import re
    pat = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.")
    idxs: set[int] = set()
    for p in bit_table:
        m = pat.match(p)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def discover_awq_groups(model, bit_table: dict[str, int], include_experts: bool) -> list[dict]:
    """Walk decoder layers (path-based, prefix-agnostic) and emit AWQ groups.

    Returns list of group dicts:
        {
          "name":          unique str id
          "members":       [{"path", "module", "bits"}, ...]
          "stats_source":  str — path whose mean_abs vector to use
          "in_dim":        int
          "absorb": {
              "target_path":  str   — module owning the weight to fold into
              "kind":         "norm_weight" | "linear_out_axis"
          }
        }

    Robust to gemma4_text ("model.layers.*") vs multimodal gemma4
    ("language_model.model.layers.*") layouts.  All paths come from
    `model.named_modules()` lookups, never attribute traversal.
    """
    layer_prefix = _detect_layer_prefix(bit_table)
    layer_idxs = _layer_indices(bit_table, layer_prefix)
    path_to_module = dict(model.named_modules())
    print(f"[group] layer prefix='{layer_prefix}', {len(layer_idxs)} decoder layers")

    groups: list[dict] = []

    def _members(prefix: str, attr_path: str, proj_names: list[str]) -> list[dict]:
        out: list[dict] = []
        for proj in proj_names:
            path = f"{prefix}.{attr_path}.{proj}"
            mod = path_to_module.get(path)
            if mod is None or path not in bit_table:
                continue
            out.append({"path": path, "module": mod, "bits": bit_table[path]})
        return out

    for i in layer_idxs:
        prefix = f"{layer_prefix}.{i}"

        # ---- G1: qkv_input ----
        members = _members(prefix, "self_attn", ["q_proj", "k_proj", "v_proj"])
        in_ln = f"{prefix}.input_layernorm"
        if members and path_to_module.get(in_ln) is not None:
            groups.append({
                "name": f"{prefix}.qkv_input",
                "members": members,
                "stats_source": members[0]["path"],
                "in_dim": int(members[0]["module"].weight.shape[-1]),
                "absorb": {"target_path": in_ln, "kind": "norm_weight"},
            })

        # ---- G2: mlp_input (Dense MLP gate+up) ----
        members = _members(prefix, "mlp", ["gate_proj", "up_proj"])
        pre_ff = f"{prefix}.pre_feedforward_layernorm"
        if members and path_to_module.get(pre_ff) is not None:
            groups.append({
                "name": f"{prefix}.mlp_input",
                "members": members,
                "stats_source": members[0]["path"],
                "in_dim": int(members[0]["module"].weight.shape[-1]),
                "absorb": {"target_path": pre_ff, "kind": "norm_weight"},
            })

        # ---- G3: mlp_down (Dense MLP down) ----
        down_path = f"{prefix}.mlp.down_proj"
        up_path = f"{prefix}.mlp.up_proj"
        down_mod = path_to_module.get(down_path)
        up_mod = path_to_module.get(up_path)
        if down_path in bit_table and down_mod is not None and up_mod is not None:
            groups.append({
                "name": f"{prefix}.mlp_down",
                "members": [{"path": down_path, "module": down_mod, "bits": bit_table[down_path]}],
                "stats_source": down_path,
                "in_dim": int(down_mod.weight.shape[-1]),
                "absorb": {"target_path": up_path, "kind": "linear_out_axis"},
            })

        # ---- Phase B: MoE expert groups (path-based, no attr traversal) ----
        if include_experts:
            e_members = _members(prefix, "experts.switch_glu", ["gate_proj", "up_proj"])
            pre_ff_2 = f"{prefix}.pre_feedforward_layernorm_2"
            if e_members and path_to_module.get(pre_ff_2) is not None:
                groups.append({
                    "name": f"{prefix}.expert_input",
                    "members": e_members,
                    "stats_source": e_members[0]["path"],
                    "in_dim": int(e_members[0]["module"].weight.shape[-1]),
                    "absorb": {"target_path": pre_ff_2, "kind": "norm_weight"},
                })

            e_down_path = f"{prefix}.experts.switch_glu.down_proj"
            e_up_path = f"{prefix}.experts.switch_glu.up_proj"
            e_down_mod = path_to_module.get(e_down_path)
            e_up_mod = path_to_module.get(e_up_path)
            if e_down_path in bit_table and e_down_mod is not None and e_up_mod is not None:
                groups.append({
                    "name": f"{prefix}.expert_down",
                    "members": [{"path": e_down_path, "module": e_down_mod, "bits": bit_table[e_down_path]}],
                    "stats_source": e_down_path,
                    "in_dim": int(e_down_mod.weight.shape[-1]),
                    "absorb": {"target_path": e_up_path, "kind": "linear_out_axis"},
                })

    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix-dir", required=True,
                    help="dir from imatrix_capture_gemma4.py (needs activation_stats.npz, "
                         "importance.npz, meta.json)")
    ap.add_argument("--src", required=True, help="bf16 source model path")
    ap.add_argument("--out", required=True,
                    help="output .npz path (sibling .json report auto-written)")
    ap.add_argument("--top4-fraction", type=float, default=0.25,
                    help="match imatrix_mixed_quant: top fraction → 4-bit; rest → 3-bit")
    ap.add_argument("--low2-fraction", type=float, default=0.0,
                    help="match imatrix_mixed_quant: bottom fraction → 2-bit (3-tier)")
    ap.add_argument("--alpha-grid", type=str,
                    default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="comma-separated α values (AWQ paper meaningful range is [0, 1])")
    ap.add_argument("--include-experts", action="store_true",
                    help="Phase B: include MoE expert SwitchLinear groups. Default: Dense only.")
    ap.add_argument("--min-improvement-pct", type=float, default=0.1,
                    help="discard AWQ scale per group when joint loss reduction below this percentage.")
    ap.add_argument("--weight-moment", type=str, default="abs", choices=["abs", "sq"],
                    help="proxy loss weighting moment. 'abs' (default, AutoAWQ style) uses E[|X|]; "
                         "'sq' uses E[X²] (sumsq/tokens), closer to true L2 reconstruction objective. "
                         "Option B in the AWQ pipeline iteration plan.")
    args = ap.parse_args()

    imatrix_dir = pathlib.Path(args.imatrix_dir)
    stats_path = imatrix_dir / "activation_stats.npz"
    importance_path = imatrix_dir / "importance.npz"
    meta_path = imatrix_dir / "meta.json"

    if not stats_path.exists():
        print(f"activation_stats.npz not found: {stats_path}\n"
              f"Re-run imatrix_capture_gemma4.py first.", file=sys.stderr)
        return 2
    if not importance_path.exists():
        print(f"importance.npz not found: {importance_path}", file=sys.stderr)
        return 2
    if not meta_path.exists():
        print(f"meta.json not found: {meta_path}", file=sys.stderr)
        return 2

    print(f"[load] activation_stats from {stats_path}")
    with np.load(stats_path) as npz:
        sumabs = {key: npz[key].astype(np.float64) for key in npz.files}
    with np.load(importance_path) as npz:
        sumsq = {key: npz[key].astype(np.float64) for key in npz.files}
    meta = json.loads(meta_path.read_text())
    token_counts = meta.get("token_counts_per_path", {})
    print(f"[load] {len(sumabs)} sumabs vectors, {len(sumsq)} sumsq vectors")

    # Build the per-input-channel weighting used by the proxy loss.
    # Option B (--weight-moment sq) approximates the true L2 reconstruction
    # objective Σ_c ε_c² · E[X_c²] more faithfully than the default mean_abs
    # weighting (which underweights heavy-tailed channels).
    weighting_np: dict[str, np.ndarray] = {}
    if args.weight_moment == "abs":
        print("[search] proxy weighting: mean_abs (E[|X|], AutoAWQ-style default)")
        for path, sa in sumabs.items():
            n = max(int(token_counts.get(path, 1)), 1)
            weighting_np[path] = (sa / n).astype(np.float32)
    else:  # "sq"
        print("[search] proxy weighting: mean_sq (E[X²], true L2 reconstruction)")
        for path, ss in sumsq.items():
            n = max(int(token_counts.get(path, 1)), 1)
            weighting_np[path] = (ss / n).astype(np.float32)
    # Backwards-compat alias used downstream — both moments populate the same
    # dict that the search loop consumes.
    mean_abs_np = weighting_np

    # Always derive `scale_input_np` from mean_abs for the actual AWQ
    # scale formula  s_c = (mean|X_c|)^α  (this matches the AWQ paper's
    # scale derivation, independent of the proxy loss weighting choice).
    scale_input_np: dict[str, np.ndarray] = {}
    for path, sa in sumabs.items():
        n = max(int(token_counts.get(path, 1)), 1)
        scale_input_np[path] = (sa / n).astype(np.float32)

    print(f"[load] model {args.src} (lazy)")
    model, _tok = load(str(args.src),
                       tokenizer_config={"eos_token": "<end_of_turn>"},
                       lazy=True)

    print("[score] reusing imatrix_mixed_quant sensitivity + bit-table ...")
    S = compute_sensitivities(model, sumsq)
    bit_table, _ranked, n_high, n_low = build_bit_table(
        S, args.top4_fraction, args.low2_fraction
    )
    n_total = len(bit_table)
    n_mid = n_total - n_high - n_low
    print(f"[plan] {n_total} quantizable tensors: "
          f"{n_high} 4-bit / {n_mid} 3-bit / {n_low} 2-bit")

    print("[group] discovering AWQ groups ...")
    groups = discover_awq_groups(model, bit_table, args.include_experts)
    n_members = sum(len(g["members"]) for g in groups)
    print(f"[group] {len(groups)} groups covering {n_members} tensors "
          f"({'Dense+experts' if args.include_experts else 'Dense only'})")

    alpha_grid = [float(a) for a in args.alpha_grid.split(",")]
    print(f"[search] alpha grid: {alpha_grid}")

    threshold = 1.0 - args.min_improvement_pct / 100.0

    scales_out: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}

    t0 = time.time()
    n_done = 0
    n_kept = 0
    n_no_stats = 0

    for g_idx, group in enumerate(groups):
        lw_np = mean_abs_np.get(group["stats_source"])
        si_np = scale_input_np.get(group["stats_source"])
        if lw_np is None or si_np is None:
            n_no_stats += 1
            report[group["name"]] = {
                "skipped_reason": f"no stats for stats_source {group['stats_source']}",
                "members": [m["path"] for m in group["members"]],
            }
            continue

        in_dim = group["in_dim"]
        if lw_np.shape[0] != in_dim or si_np.shape[0] != in_dim:
            print(f"  [skip] {group['name']}: in_dim mismatch "
                  f"(group={in_dim}, lw={lw_np.shape[0]}, si={si_np.shape[0]})",
                  file=sys.stderr)
            n_no_stats += 1
            continue

        # Resolve member W tensors (lazy materialize) — flatten SwitchLinear to 2D.
        members_with_W: list[dict] = []
        skip_group = False
        for m in group["members"]:
            W = getattr(m["module"], "weight", None)
            if W is None:
                skip_group = True
                break
            if isinstance(m["module"], SWITCH_CLASSES):
                # [E, out, in] → [E*out, in]
                E, out_dim_per_expert, _ = W.shape
                W_flat = W.reshape(E * out_dim_per_expert, in_dim)
            else:
                W_flat = W
            members_with_W.append({"path": m["path"], "W": W_flat, "bits": m["bits"]})
        if skip_group:
            continue

        loss_weight = mx.array(lw_np)
        scale_input = mx.array(si_np)

        try:
            res = search_group(
                members_with_W,
                scale_input=scale_input,
                loss_weight=loss_weight,
                alpha_grid=alpha_grid,
                group_size=GROUP_SIZE,
            )
        except Exception as e:
            print(f"  [error] {group['name']}: {e}", file=sys.stderr)
            continue

        kept = res["best_loss_total"] < res["baseline_loss_total"] * threshold

        report[group["name"]] = {
            "alpha": res["best_alpha"] if kept else 0.0,
            "improvement_pct": res["improvement_pct"],
            "baseline_loss_total": res["baseline_loss_total"],
            "best_loss_total": res["best_loss_total"],
            "kept": kept,
            "in_dim": in_dim,
            "members": [{
                "path": m["path"],
                "bits": m["bits"],
                "baseline_loss": res["per_member_baseline"][m["path"]],
                "best_loss": res["per_member_best"][m["path"]],
            } for m in group["members"]],
            "absorb": group["absorb"],
            "stats_source": group["stats_source"],
        }

        if kept:
            scales_out[group["name"]] = np.asarray(res["best_s"], dtype=np.float32)
            n_kept += 1

        # Free aggressively — bf16 26B has tight RAM.
        del members_with_W, loss_weight, scale_input
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

        n_done += 1
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(groups) - n_done) / max(rate, 1e-6)
        print(f"  [{n_done:3d}/{len(groups)}]  "
              f"α={report[group['name']]['alpha']:.2f}  "
              f"improv={res['improvement_pct']:+.2f}%  "
              f"{'KEPT ' if kept else 'noop '} "
              f"elapsed={elapsed:5.1f}s rate={rate:4.1f}/s eta={eta:5.1f}s  "
              f"{group['name']}")

    print(f"[done] processed {n_done} groups  "
          f"(kept {n_kept}, noop {n_done - n_kept}, "
          f"no-stats {n_no_stats}) in {time.time()-t0:.1f}s")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if scales_out:
        np.savez_compressed(out_path, **scales_out)
    else:
        np.savez_compressed(out_path, _empty=np.zeros(0, dtype=np.float32))
        print("[warn] no groups improved by AWQ — wrote sentinel .npz", file=sys.stderr)

    report_path = out_path.with_suffix(".json")
    report_path.write_text(json.dumps({
        "src": args.src,
        "imatrix_dir": str(imatrix_dir),
        "alpha_grid": alpha_grid,
        "top4_fraction": args.top4_fraction,
        "low2_fraction": args.low2_fraction,
        "include_experts": args.include_experts,
        "min_improvement_pct": args.min_improvement_pct,
        "weight_moment": args.weight_moment,
        "n_total_quantizable": n_total,
        "n_groups_total": len(groups),
        "n_groups_searched": n_done,
        "n_groups_kept": n_kept,
        "n_groups_no_stats": n_no_stats,
        "group_size": GROUP_SIZE,
        "groups": report,
    }, indent=2))
    print(f"[done] wrote {out_path.name} + {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
