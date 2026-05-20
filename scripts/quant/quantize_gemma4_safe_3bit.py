"""
Path A — PLE-safe / router-safe 3-bit quantization of Gemma 4 26B-A4B.

Recipe (data-free, predicate-based AFFINE-3 group_size=64):
  Quantize     : decoder nn.Linear (MoE expert SwitchLinear, q/k/v/o_proj,
                 gate/up/down_proj inside experts)
  KEEP bf16    : embed_tokens (×√hidden=53× amplifier into every layer)
                 lm_head (tied to embed_tokens — handled automatically)
                 all RMSNorms / layer scalars
                 MoE router gate (128 experts → top-K routing extremely
                 noise-sensitive)
                 vision_tower / audio_tower / cross-modal embedders

Why these layers can't take 3-bit:
  - Embedding output gets multiplied by √hidden_size (= ~53 for hidden=2816)
    so any quantization noise gets amplified by 53× into every downstream
    layer's first matmul → catastrophic in long context.
  - Router gate noise flips top-K expert selection. With 128 experts and
    sparse top-K=8, even small gate-weight perturbation steers a token to
    a completely different expert pair → drift compounds across decode steps.
  - RMSNorm scales are scalars; 3-bit affine quantization of scalars
    introduces relative error >10% with zero benefit.

Stock `mlx-community/gemma-4-26b-a4b-mlx-3bit` does NOT use this predicate —
it was made with bare `mlx_lm.convert --q-bits 3` which quantizes everything
that's divisible by group_size 64. That's the root cause of the long-context
language-drift bug v0.1.2 sampling fixed only partially.

Output is ~17-18 GB (vs broken stock 13 GB / 4-bit 22 GB).

Usage:
  python scripts/quant/quantize_gemma4_safe_3bit.py \
      --src /Users/sonheesung/models/gemma-4-26b-a4b-bf16 \
      --dst /Users/sonheesung/models/gemma-4-26b-a4b-mlx-3bit-safe

Reference: https://github.com/FakeRocket543/mlx-gemma4 (predicate inspiration)
"""

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.convert import convert as mlx_lm_convert

# Layers / weight-name fragments we MUST keep in bf16 for Gemma 4 26B-A4B.
# Order matters only for readability — `any-of` substring match.
SKIP_SUBSTR = (
    # Embeddings (×√hidden amplifier, tied with lm_head)
    "embed_tokens",
    # All normalization layers — scalars don't quantize meaningfully
    "norm",
    # MoE routing — sparse top-K is fragile
    "gate.weight",        # the linear that produces routing logits
    "router",             # alternate naming in some MoE impls
    # Multimodal towers — out of scope for text-only inference
    "vision_tower",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    # Per-layer projection-style hooks (defensive — these *may* appear if HF
    # adds them to a future Gemma 4 variant; the 26B-A4B config has them
    # disabled but the name is reserved)
    "per_layer",
    "layer_scalar",
    "embedding_post_projection",
    # `lm_head` — tied to embed_tokens in this model, so technically the
    # SKIP via embed_tokens covers it. List explicitly so a future
    # `tie_word_embeddings=false` build doesn't silently regress.
    "lm_head",
)


def make_predicate():
    """Return a `class_predicate(path, module) -> bool` for mlx_lm.utils.convert.

    Returns True for layers that SHOULD be quantized. Anything matching a
    SKIP_SUBSTR substring (case-sensitive) stays bf16.
    """

    def predicate(path: str, module) -> bool:
        # Only quantize linear-like modules. mlx_lm builds also stamp a
        # `to_quantized` attribute on already-quantized linears — skip those
        # to avoid double-quantization on reruns.
        if hasattr(module, "to_quantized") and getattr(module, "to_quantized") is False:
            return False
        is_quantizable = isinstance(module, nn.Linear) or hasattr(module, "to_quantized")
        if not is_quantizable:
            return False

        # Skip anything matching a sensitive substring.
        for s in SKIP_SUBSTR:
            if s in path:
                return False

        # MLX AFFINE-3 requires weight.size % group_size == 0. group_size=64
        # is the standard MLX choice; anything not divisible stays bf16.
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False

        return True

    return predicate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="path or HF id of the BF16 source model "
        "(e.g. /Users/sonheesung/models/gemma-4-26b-a4b-bf16)",
    )
    ap.add_argument(
        "--dst",
        required=True,
        help="output directory for the safe 3-bit MLX shards",
    )
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the would-be-quantized layer list + sizes, do not write",
    )
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr)
        return 2
    if dst.exists() and any(dst.iterdir()):
        print(
            f"dst exists and is non-empty: {dst} — remove first",
            file=sys.stderr,
        )
        return 2

    predicate = make_predicate()

    if args.dry_run:
        # Load the model with mlx-lm's loader so we can walk modules without
        # writing.  Cheaper than the full convert.
        from mlx_lm.utils import load_model

        print(f"[dry-run] loading {src} for inspection (no convert)...")
        model, _ = load_model(src, lazy=True)

        # Walk EVERY weight-carrying module, classify into:
        #   - quant: predicate matches, will be 3-bit
        #   - bf16-skipped-by-pattern: predicate rejects due to SKIP_SUBSTR
        #   - bf16-skipped-by-type: not nn.Linear / no to_quantized (e.g.
        #     nn.Embedding stays bf16 unless mlx-lm wraps it)
        #   - bf16-skipped-by-size: weight.size not divisible by group_size
        from collections import defaultdict

        kept_quant_total = 0
        skipped_total = 0
        by_skip_reason = defaultdict(int)
        skip_examples_by_substr = defaultdict(list)
        skip_examples_no_weight_attr = []
        quant_examples = []
        all_paths_with_weight = []

        for path, module in _walk_named_modules(model):
            if not hasattr(module, "weight"):
                continue
            w = module.weight
            if not hasattr(w, "size"):
                continue
            n = w.size
            all_paths_with_weight.append((path, type(module).__name__, n))

            is_quantizable = isinstance(module, nn.Linear) or hasattr(module, "to_quantized")
            if not is_quantizable:
                by_skip_reason["type-not-linear"] += n
                skip_examples_no_weight_attr.append(
                    (path, type(module).__name__, n)
                )
                continue

            matched_substr = None
            for s in SKIP_SUBSTR:
                if s in path:
                    matched_substr = s
                    break
            if matched_substr is not None:
                by_skip_reason[f"pattern:{matched_substr}"] += n
                skip_examples_by_substr[matched_substr].append((path, n))
                continue

            if n % 64 != 0:
                by_skip_reason["size%64!=0"] += n
                continue

            kept_quant_total += n
            if len(quant_examples) < 8:
                quant_examples.append((path, n))

        skipped_total = sum(by_skip_reason.values())
        total = kept_quant_total + skipped_total
        print(f"  total weight params  : {total/1e9:.2f} B")
        print(
            f"  → quantized (3-bit)  : {kept_quant_total/1e9:.2f} B  "
            f"({100*kept_quant_total/total:.1f}%)"
        )
        print(
            f"  → kept bf16          : {skipped_total/1e9:.2f} B  "
            f"({100*skipped_total/total:.1f}%)"
        )
        print("\n  bf16-skip breakdown by reason:")
        for reason, n_params in sorted(
            by_skip_reason.items(), key=lambda kv: -kv[1]
        ):
            print(f"    {reason:30s}  {n_params/1e6:10.1f} M")

        print("\n  bf16-skip examples per SKIP_SUBSTR pattern:")
        for substr, examples in skip_examples_by_substr.items():
            total_n = sum(n for _, n in examples)
            print(
                f"    '{substr}' → {len(examples)} layers, "
                f"{total_n/1e6:.1f} M params total"
            )
            for path, n in examples[:3]:
                print(f"        {path:60s}  {n/1e6:8.1f} M")

        if skip_examples_no_weight_attr:
            print("\n  bf16 by non-Linear type (sample):")
            for path, tname, n in skip_examples_no_weight_attr[:8]:
                print(f"    {path:50s}  [{tname}]  {n/1e6:.1f} M")

        print("\n  sample QUANT (3-bit):")
        for path, n in quant_examples:
            print(f"    {path:60s}  {n/1e6:8.1f} M")
        print(
            "\n  estimated output size: "
            f"~{(skipped_total*2 + kept_quant_total*3/8)/1e9:.1f} GB"
        )
        return 0

    # NOTE: do NOT pre-create dst; mlx_lm_convert refuses if the path
    # already exists (even if empty). It creates the dir itself.
    print(
        f"[convert] {src} → {dst}  (q={args.bits}, gs={args.group_size}, "
        f"predicate=PLE/router-safe)"
    )
    # mlx_lm.utils.convert is the canonical entry point. quant_predicate
    # is a callable matching the same `(path, module) -> bool` shape.
    mlx_lm_convert(
        hf_path=str(src),
        mlx_path=str(dst),
        quantize=True,
        q_group_size=args.group_size,
        q_bits=args.bits,
        dtype="bfloat16",
        quant_predicate=predicate,
    )

    # Drop a marker so we know *which* predicate produced this build.
    marker = dst / "lumen_quant_recipe.json"
    marker.write_text(
        json.dumps(
            {
                "recipe": "lumen.gemma4.safe_3bit",
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_substr": list(SKIP_SUBSTR),
                "source": str(src),
            },
            indent=2,
        )
    )
    print(f"[convert] wrote {marker.name}")
    return 0


def _walk_named_modules(root):
    """Yield `(path, module)` pairs for every submodule.

    Uses mlx.nn.Module.named_modules() which returns a dict-iterator of
    `(path, module)` tuples. Filtering out the empty-path root for clarity.
    """
    for path, module in root.named_modules():
        if path == "":
            continue
        yield path, module


if __name__ == "__main__":
    raise SystemExit(main())
