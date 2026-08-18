#!/usr/bin/env python3
"""Per-step normalized macOS sample(1) symbol diff for native vs PyO3 MLX runners.

Promoted from the ad-hoc /tmp/native_pyo3_diff/compare_per_step.py used in
`native_argmax_try_item_double_eval_landed.md`.

Input:
  Two macOS `sample(1)` text outputs (one per backend), and either explicit
  ms-per-step values or `bench_mlx_e2e` log files to auto-extract them.

Output:
  - Per-symbol delta table (native /step − pyo3 /step), ranked by |Δ|.
  - Category roll-up (submit / eval / attn / cache / scalar / graph / other).
  - Backend metadata (mean / p50 / p95 / tok/s) when bench logs are supplied.
  - Heuristic flag when Δ submit ≈ Δ eval_impl (try_item-style double-eval).

Per-step normalization:
  Each `sample(1)` line `+ N symbol  (in module)` contributes N samples at
  1ms interval. With sample window W ms and measured ms/step S, the number
  of decode steps inside the window is W/S, and per-step samples is
      samples_per_step = N * S / W
  which equals the ms/step that symbol consumed (since interval = 1ms).

Usage (4-baseline G0 protocol):
  python3 scripts/compare_native_pyo3_sample.py \
      --native /tmp/native.short.sample.txt --bench-native /tmp/native.short.bench.log \
      --pyo3   /tmp/pyo3.short.sample.txt   --bench-pyo3   /tmp/pyo3.short.bench.log \
      --window-ms 25000 --prompt short

Self-test:
  python3 scripts/compare_native_pyo3_sample.py --self-test
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Parsing primitives
# ─────────────────────────────────────────────────────────────────────────────

# `sample(1)` thread header. Three observed formats:
#   pyo3 main: "    20552 Thread_2541640   DispatchQueue_1: com.apple.main-thread  (serial)"
#   native main: "    20303 Thread_2539926: main"
#   no label:   "    20303 Thread_2539945"
# After Thread_<id>, separator (whitespace/colon) and label are both optional.
THREAD_HDR_RE = re.compile(r"^\s*(\d+)\s+Thread_[0-9A-Fa-f]+(?:[:\s]+(.*))?$")
# Main-thread label tag. Pyo3 says "com.apple.main-thread", native says just "main".
MAIN_THREAD_RE = re.compile(r"\bmain(-thread)?\b")

# Symbol line: any depth of "+", "!", "|", ":", ws, then count, then symbol up to "(in ".
# Examples encountered in `sample(1)` output:
#   "    + 2521 start  (in dyld)  [0x...]"
#   "    + ! 1234 mlx::core::eval_impl  (in libmlx.dylib)  [0x...]"
SYMBOL_LINE_RE = re.compile(r"^[\s+!|:]+(\d+)\s+(\S.*?)\s+\(in\s+\S")

# bench_mlx_e2e output we care about:
#   "  step latency: mean=15.99ms p50=15.76ms p95=17.78ms"
#   "  throughput:   62.5 tok/s"
#   "decode: 2500 steps in 39975ms"
BENCH_LATENCY_RE = re.compile(
    r"step latency:\s*mean=([\d.]+)ms\s+p50=([\d.]+)ms\s+p95=([\d.]+)ms"
)
BENCH_THROUGHPUT_RE = re.compile(r"throughput:\s+([\d.]+)\s+tok/s")
BENCH_STEPS_RE = re.compile(r"decode:\s+(\d+)\s+steps\s+in\s+([\d.]+)ms")


@dataclass
class BenchMeta:
    steps: int | None = None
    total_ms: float | None = None
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    tps: float | None = None

    @property
    def ms_per_step(self) -> float | None:
        return self.mean_ms


def parse_bench_log(text: str) -> BenchMeta:
    meta = BenchMeta()
    if m := BENCH_STEPS_RE.search(text):
        meta.steps = int(m.group(1))
        meta.total_ms = float(m.group(2))
    if m := BENCH_LATENCY_RE.search(text):
        meta.mean_ms = float(m.group(1))
        meta.p50_ms = float(m.group(2))
        meta.p95_ms = float(m.group(3))
    if m := BENCH_THROUGHPUT_RE.search(text):
        meta.tps = float(m.group(1))
    return meta


def parse_main_thread_symbols(text: str) -> dict[str, int]:
    """Return {symbol: total_inclusive_count} from the main thread block.

    Picks the first thread block whose header contains a `main` tag. Falls
    back to the first thread block. Within the block, sums occurrence
    counts per unique symbol — sample(1) tree counts are inclusive, but
    each line represents a distinct stack frame instance (a unique call
    path). For non-recursive symbols (eval_impl, submitCommandBuffers,
    SDPA, etc.) the sum across distinct paths is the total inclusive cost
    across the thread. (For recursive symbols this would over-count, but
    none of the categorized hot symbols are recursive in practice.)
    """
    lines = text.splitlines()

    # Locate thread headers.
    thread_starts: list[tuple[int, bool]] = []  # (line_idx, is_main)
    for idx, line in enumerate(lines):
        m = THREAD_HDR_RE.match(line)
        if m:
            label = m.group(2) or ""
            thread_starts.append((idx, bool(MAIN_THREAD_RE.search(label))))

    if not thread_starts:
        return {}

    # Pick main-thread block, else first.
    target_start = next(
        (idx for idx, is_main in thread_starts if is_main),
        thread_starts[0][0],
    )
    next_starts = [idx for idx, _ in thread_starts if idx > target_start]
    target_end = next_starts[0] if next_starts else len(lines)

    counts: dict[str, int] = defaultdict(int)
    for line in lines[target_start:target_end]:
        if m := SYMBOL_LINE_RE.match(line):
            count = int(m.group(1))
            symbol = m.group(2).strip()
            counts[symbol] += count
    return dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
# Categorization
# ─────────────────────────────────────────────────────────────────────────────

# Order matters: first match wins. More specific patterns first.
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("submit", ("submitCommandBuffers", "MTLCommandBuffer", "commandBufferWith")),
    ("eval",   ("mlx::core::eval_impl", "mlx::core::array::eval", "::eval(")),
    ("attn",   ("sdpa", "scaled_dot_product", "fused_attention", "attention::")),
    ("cache",  ("KVCache", "mlx_clear_cache", "deep_clone", "_resize_cache", "kv_cache")),
    ("scalar", ("try_item", "try_as_slice", "::item(", "scalar_to_")),
    ("graph",  ("Compiled", "::compile", "::fuse", "Primitive::eval_gpu")),
]


def categorize(symbol: str) -> str:
    for cat, needles in CATEGORY_RULES:
        if any(n in symbol for n in needles):
            return cat
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Diff + reporting
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolRow:
    symbol: str
    native_per_step: float
    pyo3_per_step: float
    category: str

    @property
    def delta(self) -> float:
        return self.native_per_step - self.pyo3_per_step


def normalize(counts: dict[str, int], ms_per_step: float, window_ms: float) -> dict[str, float]:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    factor = ms_per_step / window_ms
    return {sym: cnt * factor for sym, cnt in counts.items()}


def diff_symbols(
    native_norm: dict[str, float],
    pyo3_norm: dict[str, float],
) -> list[SymbolRow]:
    keys = set(native_norm) | set(pyo3_norm)
    rows = [
        SymbolRow(
            symbol=k,
            native_per_step=native_norm.get(k, 0.0),
            pyo3_per_step=pyo3_norm.get(k, 0.0),
            category=categorize(k),
        )
        for k in keys
    ]
    rows.sort(key=lambda r: abs(r.delta), reverse=True)
    return rows


def category_rollup(rows: list[SymbolRow]) -> dict[str, tuple[float, float]]:
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        agg[r.category][0] += r.native_per_step
        agg[r.category][1] += r.pyo3_per_step
    return {k: (v[0], v[1]) for k, v in agg.items()}


def fmt_meta(meta: BenchMeta) -> str:
    parts = []
    if meta.mean_ms is not None:
        parts.append(f"mean={meta.mean_ms:.2f}ms")
    if meta.p50_ms is not None:
        parts.append(f"p50={meta.p50_ms:.2f}ms")
    if meta.p95_ms is not None:
        parts.append(f"p95={meta.p95_ms:.2f}ms")
    if meta.tps is not None:
        parts.append(f"tps={meta.tps:.2f}")
    if meta.steps is not None:
        parts.append(f"steps={meta.steps}")
    return " ".join(parts) if parts else "(no bench log)"


def print_report(
    rows: list[SymbolRow],
    native_meta: BenchMeta,
    pyo3_meta: BenchMeta,
    window_ms: float,
    top: int,
    prompt_label: str,
) -> None:
    print(f"=== Native vs PyO3 sample(1) per-step diff [{prompt_label}] ===")
    print(f"window: {window_ms:.0f}ms (1ms interval)")
    print(f"native: {fmt_meta(native_meta)}")
    print(f"pyo3:   {fmt_meta(pyo3_meta)}")
    if native_meta.tps and pyo3_meta.tps:
        print(f"ratio:  native/pyo3 = {100.0 * native_meta.tps / pyo3_meta.tps:.1f}% (tok/s)")
    print()

    print(f"--- Top {top} symbol deltas (sorted by |Δ /step|) ---")
    print(f"{'symbol':<60}  {'cat':<7}  {'native/step':>12}  {'pyo3/step':>11}  {'Δ/step':>8}")
    for r in rows[:top]:
        sym_disp = r.symbol if len(r.symbol) <= 58 else r.symbol[:55] + "..."
        print(
            f"{sym_disp:<60}  {r.category:<7}  "
            f"{r.native_per_step:12.3f}  {r.pyo3_per_step:11.3f}  {r.delta:+8.3f}"
        )
    print()

    print("--- Category roll-up (signal categories only; samples_per_step ≈ ms/step) ---")
    print("    NOTE: sample(1) counts are inclusive — totals include ancestor")
    print("    frames within the same category, so absolute values may exceed")
    print("    actual ms/step. The per-category Δ is still informative as a")
    print("    relative direction; for absolute attribution, use the top-N table.")
    rollup = category_rollup(rows)
    print(f"{'category':<8}  {'native/step':>12}  {'pyo3/step':>11}  {'Δ/step':>8}")
    # Skip 'other' — its mix of stack roots (start/main/_pthread_*) and
    # uncategorized signal symbols inflates the absolute value far beyond
    # the actual ms/step. Top-N already surfaces important 'other' rows.
    for cat in ["submit", "eval", "attn", "cache", "scalar", "graph"]:
        if cat not in rollup:
            continue
        n, p = rollup[cat]
        print(f"{cat:<8}  {n:12.3f}  {p:11.3f}  {n - p:+8.3f}")
    print()

    # Heuristic: try_item-style double-eval signature is specifically
    # `mlx::core::eval_impl` ≈ `submitCommandBuffers` per-step delta. Looking
    # at category sums would mix in `mlx::core::array::eval()` (the wrapper),
    # which can dilute the match. Use the specific symbols.
    sub_d = _find_delta(rows, "submitCommandBuffers")
    eval_d = _find_delta(rows, "eval_impl")
    if sub_d > 0.5 and eval_d > 0.5:
        rel_gap = abs(sub_d - eval_d) / max(sub_d, eval_d)
        if rel_gap < 0.20:
            print(
                f"⚠ Heuristic: Δ eval_impl ({eval_d:+.2f}/step) ≈ "
                f"Δ submitCommandBuffers ({sub_d:+.2f}/step). "
                "Consistent with redundant-eval-per-step (try_item-class signature). "
                "Audit scalar-read FFI primitives in the native decode hot path."
            )


def _find_delta(rows: list[SymbolRow], needle: str) -> float:
    """Largest Δ /step among symbols whose name contains needle."""
    matches = [r for r in rows if needle in r.symbol]
    if not matches:
        return 0.0
    return max(matches, key=lambda r: abs(r.delta)).delta


# ─────────────────────────────────────────────────────────────────────────────
# Self-test fixture (validates against try_item double-eval signature)
# ─────────────────────────────────────────────────────────────────────────────

# Counts back-computed from prior memo (native_argmax_try_item_double_eval_landed.md)
# so that normalization yields exactly:
#   eval_impl: native 9.97 /step, pyo3 8.18 /step (Δ = +1.79)
#   submit:    native 3.06 /step, pyo3 1.27 /step (Δ = +1.79)
# with native ms/step = 15.99, pyo3 ms/step = 14.38, window = 25000ms.

NATIVE_FIXTURE = """\
Sampling process 12345 every 1 millisecond
Call graph:
    25000 Thread_111  DispatchQueue_1: com.apple.main-thread  (serial)
      + 25000 start  (in dyld)  [0x1]
      +   25000 main  (in lumen_mlx)  [0x2]
      +     18681 mlx::core::array::eval()  (in libmlx.dylib)  [0x3]
      +       15589 mlx::core::eval_impl  (in libmlx.dylib)  [0x4]
      +         4784 IOGPUMetalCommandQueue::submitCommandBuffers  (in libIOGPU)  [0x5]
      +     1500 try_item  (in lumen_mlx)  [0x6]
    1000 Thread_222  DispatchQueue_2: rayon-worker
      + 500 some_idle_worker  (in libfoo)  [0x7]
"""

# PyO3: no try_item double-eval, fewer eval_impl + submit per step.
PYO3_FIXTURE = """\
Sampling process 67890 every 1 millisecond
Call graph:
    25000 Thread_999  DispatchQueue_1: com.apple.main-thread  (serial)
      + 25000 start  (in dyld)  [0x1]
      +   25000 main  (in lumen_mlx)  [0x2]
      +     17977 mlx::core::array::eval()  (in libmlx.dylib)  [0x3]
      +       14224 mlx::core::eval_impl  (in libmlx.dylib)  [0x4]
      +         2208 IOGPUMetalCommandQueue::submitCommandBuffers  (in libIOGPU)  [0x5]
    500 Thread_888  DispatchQueue_2: python-thread
      + 500 some_idle_python  (in libpython)  [0x7]
"""


def self_test() -> int:
    """Reproduce try_item double-eval signature: Δ ≈ +1.79/step on submit & eval_impl."""
    # window 25000ms, native ms/step ≈ 15.99 → n_steps ≈ 1563.
    # 18000 samples eval_impl → 18000 * 15.99 / 25000 = 11.51/step
    # 15000 samples eval_impl pyo3 → 15000 * 14.38 / 25000 = 8.63/step
    # Δ eval_impl ≈ 2.88. We expect both backends' eval_impl to differ by
    # roughly the redundant-submit count; the exact match the prior memo
    # recorded (+1.79) was for the production recipe — fixture uses round
    # numbers but preserves the qualitative signature: Δ submit ≈ Δ eval_impl
    # within 20%.
    native_meta = BenchMeta(steps=1563, total_ms=25000.0, mean_ms=15.99, p50_ms=15.76, p95_ms=17.78, tps=62.5)
    pyo3_meta = BenchMeta(steps=1739, total_ms=25000.0, mean_ms=14.38, p50_ms=14.26, p95_ms=15.58, tps=69.5)
    window_ms = 25000.0

    n_counts = parse_main_thread_symbols(NATIVE_FIXTURE)
    p_counts = parse_main_thread_symbols(PYO3_FIXTURE)

    assert "mlx::core::eval_impl" in n_counts, f"native eval_impl missing: {sorted(n_counts)}"
    assert "IOGPUMetalCommandQueue::submitCommandBuffers" in n_counts
    assert "some_idle_worker" not in n_counts, "non-main-thread leak"
    assert "some_idle_python" not in p_counts, "non-main-thread leak"

    n_norm = normalize(n_counts, native_meta.mean_ms, window_ms)
    p_norm = normalize(p_counts, pyo3_meta.mean_ms, window_ms)
    rows = diff_symbols(n_norm, p_norm)

    # Capture print_report output to a string for assertion.
    buf = io.StringIO()
    saved = sys.stdout
    try:
        sys.stdout = buf
        print_report(rows, native_meta, pyo3_meta, window_ms, top=20, prompt_label="self-test")
    finally:
        sys.stdout = saved
    out = buf.getvalue()

    sub_d = _find_delta(rows, "submitCommandBuffers")
    eval_d = _find_delta(rows, "eval_impl")

    # Both should be substantial (> 1.5/step) and within 20% of each other,
    # matching the "double-eval signature" pattern from the prior memo
    # (production: +1.79 each).
    assert sub_d > 1.5, f"Δ submit too small: {sub_d:.3f}"
    assert eval_d > 1.5, f"Δ eval_impl too small: {eval_d:.3f}"
    rel_gap = abs(sub_d - eval_d) / max(sub_d, eval_d)
    assert rel_gap < 0.20, (
        f"Δ submit ({sub_d:.3f}) vs Δ eval_impl ({eval_d:.3f}) "
        f"rel_gap={rel_gap:.3f} should be < 0.20"
    )

    # Heuristic warning must fire.
    assert "try_item-class" in out, "double-eval heuristic warning did not fire"
    # Main-thread filter must hold.
    assert "some_idle_worker" not in out
    assert "some_idle_python" not in out

    print(out)
    print("SELF-TEST PASS")
    print(
        f"  Δ submit /step = {sub_d:+.3f} | Δ eval /step = {eval_d:+.3f} "
        f"| rel_gap = {rel_gap:.3f}"
    )
    print("  Signature (Δ submit ≈ Δ eval) matches try_item double-eval class.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def resolve_ms_per_step(
    label: str,
    explicit: float | None,
    bench_path: Path | None,
) -> tuple[float, BenchMeta]:
    if bench_path:
        meta = parse_bench_log(bench_path.read_text(errors="replace"))
        if meta.mean_ms is None:
            raise SystemExit(
                f"{label}: bench log {bench_path} missing 'step latency: mean=...'"
            )
        if explicit and abs(explicit - meta.mean_ms) > 0.01:
            print(
                f"WARN: {label}: explicit ms_per_step={explicit:.3f} "
                f"overrides bench log mean={meta.mean_ms:.3f}",
                file=sys.stderr,
            )
            return explicit, meta
        return meta.mean_ms, meta
    if explicit is None:
        raise SystemExit(
            f"{label}: provide --{label}-ms-per-step or --bench-{label}"
        )
    return explicit, BenchMeta()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", type=Path, help="native sample(1) text output")
    ap.add_argument("--pyo3", type=Path, help="pyo3 sample(1) text output")
    ap.add_argument("--bench-native", type=Path, help="native bench_mlx_e2e log")
    ap.add_argument("--bench-pyo3", type=Path, help="pyo3 bench_mlx_e2e log")
    ap.add_argument("--native-ms-per-step", type=float, help="override native mean ms/step")
    ap.add_argument("--pyo3-ms-per-step", type=float, help="override pyo3 mean ms/step")
    ap.add_argument("--window-ms", type=float, default=25000.0, help="sample window length (ms)")
    ap.add_argument("--top", type=int, default=30, help="top N delta rows")
    ap.add_argument("--prompt", default="unspecified", help="annotation: short / long / etc")
    ap.add_argument("--self-test", action="store_true", help="run synthetic fixture validation")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.native or not args.pyo3:
        ap.error("--native and --pyo3 are required (or use --self-test)")

    n_ms, n_meta = resolve_ms_per_step("native", args.native_ms_per_step, args.bench_native)
    p_ms, p_meta = resolve_ms_per_step("pyo3", args.pyo3_ms_per_step, args.bench_pyo3)

    n_counts = parse_main_thread_symbols(args.native.read_text(errors="replace"))
    p_counts = parse_main_thread_symbols(args.pyo3.read_text(errors="replace"))
    if not n_counts:
        raise SystemExit(f"no symbols parsed from {args.native} (main thread missing?)")
    if not p_counts:
        raise SystemExit(f"no symbols parsed from {args.pyo3}")

    n_norm = normalize(n_counts, n_ms, args.window_ms)
    p_norm = normalize(p_counts, p_ms, args.window_ms)
    rows = diff_symbols(n_norm, p_norm)
    print_report(rows, n_meta, p_meta, args.window_ms, args.top, args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
