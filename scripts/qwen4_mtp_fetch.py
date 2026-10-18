#!/usr/bin/env python3
"""Fetch the MTP (multi-token-prediction) draft head of Qwen3.8-Flash-Next from a public
MLX 4-bit repo and write it as `<model_dir>/mtp.safetensors` (+ `<model_dir>/mtp.json`).

The repo (default `rapid-mlx/Qwen3.8-Flash-Next-4bit`) keeps the `mtp.*` tensors in its last two
shards (27 and 28, ~1.47 GB of a ~100 GB checkpoint). Nothing but those bytes is downloaded:
the index (`model.safetensors.index.json`) names the shards, each shard's safetensors header
(8-byte little-endian length + JSON) gives the byte ranges, and the tensors are pulled with HTTP
Range requests (no auth, redirects followed, transient errors retried). Only the Python standard
library and numpy are needed.

Transforms applied while writing (everything else keeps its name, dtype, shape and bytes):

  1. FUSED ROUTED EXPERTS ARE SPLIT.  The repo stores
         mtp.layers.0.mlp.experts.gate_up_proj{,.scales,.biases}   [E, 2*I, X]
         mtp.layers.0.mlp.experts.down_proj{,.scales,.biases}      [E, H,   X]
     (the weight tensors carry NO `.weight` suffix). They are written as the stacked tensors
         mtp.layers.0.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
     gate_up is cut along axis 1 into two halves of I rows:

         >>> FIRST half = gate_proj, SECOND half = up_proj  <<<

     That ordering is an ASSUMPTION: it follows the HF/transformers convention
     `gate, up = gate_up.chunk(2, dim=...)` (and Slotstream's converter / the vendored reference's
     `sanitize`, which both take gate first). Nothing in the file proves it. If the draft's accept
     rate is near zero while everything else checks out, swap the halves (`gate_up_order` in the
     safetensors metadata and in mtp.json records which one was written: "gate_first").

  2. NORM WEIGHTS GET +1 FOLDED IN.  The repo stores these RMSNorm weights zero-centered (the HF
     reference computes x/rms * (1 + w)); the model code multiplies by w, so +1 is added once
     here, in float32, and rounded to nearest-even back to bf16. Exactly the tensors whose name
     ends in one of
         hc_norm.weight  q_norm.weight  k_norm.weight  q_layernorm.weight  k_layernorm.weight
         pre_fc_norm_embedding.weight  pre_fc_norm_hidden.weight
     (for the MTP block: 3x hc_norm, self_attn q_norm/k_norm, indexer q/k_layernorm and the two
     pre_fc norms = 9 tensors). Slotstream's Tools/mtp_convert.py folds the same set (its CENTERED
     tuple spells the indexer ones `indexer.q_layernorm.weight`), with
     `(v.astype(float32) + 1.0).astype(bfloat16)`, i.e. the same rounding.

Everything is group size 64; `mtp.json` records the bit width of every quantized linear / stacked
expert tensor, derived from shapes: bits = weight_cols * 32 / (scales_cols * 64).

usage: python3 scripts/qwen4_mtp_fetch.py <model_dir> [--repo OWNER/NAME] [--revision REV] [--force]
"""
from __future__ import annotations

import argparse
import bisect
import http.client
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

DEFAULT_REPO = "rapid-mlx/Qwen3.8-Flash-Next-4bit"
GROUP_SIZE = 64
CHUNK = 32 << 20  # bytes per range request
GAP_MERGE = 64 << 10  # fetch across gaps up to this size rather than issuing another request
ITEMSIZE = {"U32": 4, "BF16": 2}

# Norm weights stored zero-centered in the repo; +1 is folded in (see module docstring).
CENTERED_SUFFIXES = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)

_FUSED = re.compile(r"^(?P<base>.*)\.experts\.(?P<kind>gate_up_proj|down_proj)(?P<suffix>\.scales|\.biases)?$")

_UA = {"User-Agent": "qwen4_mtp_fetch/1"}


# ------------------------------------------------------------------------------- http


def _sleep_for(attempt: int, retry_after: str | None) -> None:
    delay = min(60.0, 2.0 * (2**attempt))
    if retry_after and retry_after.isdigit():
        delay = max(delay, float(retry_after))
    time.sleep(delay)


def http_get(url: str, rng: tuple[int, int] | None = None, tries: int = 7) -> tuple[bytes, int | None]:
    """GET (optionally an inclusive byte range). Returns (body, total_size_from_Content-Range).

    Range responses must be 206 with exactly the requested length; transient failures (429,
    5xx, resets, short reads, timeouts) are retried with exponential backoff.
    """
    headers = dict(_UA)
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    last: Exception | None = None
    for attempt in range(tries):
        retry_after = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                body = r.read()
                total = None
                if rng is not None:
                    if r.status != 206:
                        raise RuntimeError(f"server ignored the Range header (status {r.status}) for {url}")
                    cr = r.headers.get("Content-Range", "")
                    m = re.match(r"bytes \d+-\d+/(\d+)$", cr)
                    total = int(m.group(1)) if m else None
                    want = rng[1] - rng[0] + 1
                    if len(body) != want:
                        raise http.client.IncompleteRead(body, want - len(body))
                return body, total
        except urllib.error.HTTPError as e:
            if e.code not in (408, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {e.code} for {url}") from e
            last, retry_after = e, e.headers.get("Retry-After")
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout, TimeoutError) as e:
            last = e
        if attempt + 1 < tries:
            _sleep_for(attempt, retry_after)
    raise RuntimeError(f"giving up on {url} range={rng}: {last!r}")


# ------------------------------------------------------------------------------- plan


@dataclass
class OutTensor:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int
    # (shard, absolute source offset, length, offset inside this tensor)
    pieces: list[tuple[str, int, int, int]] = field(default_factory=list)
    offset: int = 0  # relative to the start of the data section


def read_shard_header(base_url: str, shard: str) -> tuple[dict, int, int]:
    """(tensor table, absolute offset of the data section, file size)."""
    raw, total = http_get(f"{base_url}/{shard}", (0, 7))
    (n,) = struct.unpack("<Q", raw)
    if n > 64 << 20:
        raise RuntimeError(f"{shard}: implausible header length {n}")
    hdr, _ = http_get(f"{base_url}/{shard}", (8, 8 + n - 1))
    return json.loads(hdr), 8 + n, int(total)


def build_plan(base_url: str) -> list[OutTensor]:
    index, _ = http_get(f"{base_url}/model.safetensors.index.json")
    weight_map = json.loads(index)["weight_map"]
    mtp = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}
    if not mtp:
        raise RuntimeError("no mtp.* tensors in the repo's index")
    shards = sorted(set(mtp.values()))
    print(f"{len(mtp)} mtp.* tensors in {len(shards)} shards: {', '.join(shards)}")

    headers = {s: read_shard_header(base_url, s) for s in shards}
    outs: dict[str, OutTensor] = {}

    def add(t: OutTensor) -> None:
        if t.name in outs:
            raise RuntimeError(f"duplicate output tensor {t.name}")
        outs[t.name] = t

    for name, shard in sorted(mtp.items()):
        hdr, data_start, fsize = headers[shard]
        e = hdr[name]
        dtype, shape = e["dtype"], list(e["shape"])
        if dtype not in ITEMSIZE:
            raise RuntimeError(f"{name}: unexpected dtype {dtype}")
        lo, hi = e["data_offsets"]
        n = 1
        for d in shape:
            n *= d
        if hi - lo != n * ITEMSIZE[dtype] or data_start + hi > fsize:
            raise RuntimeError(f"{name}: header offsets {lo}..{hi} do not match shape {shape} / file size {fsize}")
        src = data_start + lo

        m = _FUSED.match(name)
        if not m:
            add(OutTensor(name, dtype, shape, hi - lo, [(shard, src, hi - lo, 0)]))
            continue

        suffix = {None: ".weight", ".scales": ".scales", ".biases": ".biases"}[m.group("suffix")]
        pre = f"{m.group('base')}.switch_mlp."
        if m.group("kind") == "down_proj":
            add(OutTensor(f"{pre}down_proj{suffix}", dtype, shape, hi - lo, [(shard, src, hi - lo, 0)]))
            continue

        # gate_up_proj: [E, 2*I, X] -> gate_proj [E, I, X] (first half) + up_proj [E, I, X] (second half)
        n_exp, rows = shape[0], shape[1]
        if rows % 2:
            raise RuntimeError(f"{name}: odd fused row count {rows}")
        per_expert = (hi - lo) // n_exp
        half = per_expert // 2
        half_shape = [n_exp, rows // 2] + shape[2:]
        gate = OutTensor(f"{pre}gate_proj{suffix}", dtype, half_shape, n_exp * half)
        up = OutTensor(f"{pre}up_proj{suffix}", dtype, half_shape, n_exp * half)
        for x in range(n_exp):
            gate.pieces.append((shard, src + x * per_expert, half, x * half))
            up.pieces.append((shard, src + x * per_expert + half, half, x * half))
        add(gate)
        add(up)

    # layout: sorted by name, data section starts at 0
    off = 0
    for name in sorted(outs):
        outs[name].offset = off
        off += outs[name].nbytes
    return [outs[k] for k in sorted(outs)]


# --------------------------------------------------------------------- bf16 helpers


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bytes(f: np.ndarray) -> bytes:
    """float32 -> bf16 with round-to-nearest-even (NaN stays NaN)."""
    bits = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    rounded = np.where(np.isnan(f), (bits >> 16) | 0x40, rounded)
    return rounded.astype("<u2").tobytes()


def fold_one(raw: bytes) -> bytes:
    return f32_to_bf16_bytes(bf16_to_f32(raw) + np.float32(1.0))


# ------------------------------------------------------------------------- transfer


class Progress:
    def __init__(self, total: int) -> None:
        self.total, self.done, self.t0, self.last = total, 0, time.time(), 0.0
        self.lock = threading.Lock()
        self.tty = sys.stderr.isatty()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last < (0.5 if self.tty else 5.0) and self.done < self.total:
                return
            self.last = now
            rate = self.done / max(now - self.t0, 1e-6) / 1e6
            line = f"  {self.done / 1e6:8.1f} / {self.total / 1e6:.1f} MB  ({100 * self.done / self.total:5.1f}%)  {rate:6.1f} MB/s"
            print("\r" + line if self.tty else line, end="" if self.tty else "\n", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr)


def transfer(base_url: str, plan: list[OutTensor], fd: int, data_start: int, workers: int) -> int:
    """Download every piece and scatter it to its place in the output file. Returns bytes written."""
    per_shard: dict[str, list[tuple[int, int, int]]] = {}
    for t in plan:
        for shard, src, length, rel in t.pieces:
            per_shard.setdefault(shard, []).append((src, length, data_start + t.offset + rel))
    total = sum(t.nbytes for t in plan)
    written = [0]
    wlock = threading.Lock()
    progress = Progress(total)

    jobs: list[tuple[str, int, int, list[int], list[tuple[int, int, int]]]] = []
    for shard, ops in per_shard.items():
        ops.sort()
        starts = [o[0] for o in ops]
        i = 0
        while i < len(ops):
            a = ops[i][0]
            end = ops[i][0] + ops[i][1]
            j = i + 1
            while j < len(ops) and ops[j][0] <= end + GAP_MERGE:
                end = max(end, ops[j][0] + ops[j][1])
                j += 1
            for lo in range(a, end, CHUNK):
                jobs.append((shard, lo, min(lo + CHUNK, end), starts, ops))
            i = j

    def run(job) -> None:
        shard, a, b, starts, ops = job
        body, _ = http_get(f"{base_url}/{shard}", (a, b - 1))
        mv = memoryview(body)
        k = max(bisect.bisect_right(starts, a) - 1, 0)
        n = 0
        while k < len(ops) and ops[k][0] < b:
            s, length, dst = ops[k]
            lo, hi = max(a, s), min(b, s + length)
            if lo < hi:
                os.pwrite(fd, mv[lo - a : hi - a], dst + (lo - s))
                n += hi - lo
            k += 1
        with wlock:
            written[0] += n
        progress.add(len(body))

    print(f"downloading {total / 1e6:.1f} MB in {len(jobs)} range requests ({workers} workers)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(run, j) for j in jobs]):
            f.result()
    progress.finish()
    if written[0] != total:
        raise RuntimeError(f"wrote {written[0]} bytes, expected {total}")
    return written[0]


# ------------------------------------------------------------------------- output


def header_bytes(plan: list[OutTensor], metadata: dict[str, str]) -> bytes:
    hdr: dict = {"__metadata__": metadata}
    for t in plan:
        hdr[t.name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [t.offset, t.offset + t.nbytes]}
    s = json.dumps(hdr, separators=(",", ":")).encode()
    s += b" " * (-len(s) % 8)
    return struct.pack("<Q", len(s)) + s


def derive_bits(plan: list[OutTensor]) -> dict[str, int]:
    by = {t.name: t for t in plan}
    bits: dict[str, int] = {}
    for t in plan:
        if t.dtype != "U32":
            continue
        if not t.name.endswith(".weight"):
            raise RuntimeError(f"{t.name}: U32 tensor without .weight suffix")
        base = t.name[: -len(".weight")]
        sc, bi = by.get(base + ".scales"), by.get(base + ".biases")
        if sc is None or bi is None:
            raise RuntimeError(f"{t.name}: missing .scales/.biases")
        if sc.shape != bi.shape or sc.dtype != "BF16" or bi.dtype != "BF16":
            raise RuntimeError(f"{base}: scales/biases mismatch {sc.shape}/{bi.shape}")
        if sc.shape[:-1] != t.shape[:-1]:
            raise RuntimeError(f"{base}: scales {sc.shape} vs weight {t.shape}")
        num = t.shape[-1] * 32
        den = sc.shape[-1] * GROUP_SIZE
        if num % den or num // den not in (2, 3, 4, 6, 8):
            raise RuntimeError(f"{base}: cannot derive bits from weight {t.shape} scales {sc.shape} (group {GROUP_SIZE})")
        bits[base] = num // den
    return bits


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="overwrite an existing mtp.safetensors")
    args = ap.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    out_path = os.path.join(model_dir, "mtp.safetensors")
    json_path = os.path.join(model_dir, "mtp.json")
    if not os.path.isdir(model_dir):
        print(f"error: {model_dir} is not a directory", file=sys.stderr)
        return 2
    if os.path.exists(out_path) and not args.force:
        print(f"error: {out_path} exists (use --force to overwrite)", file=sys.stderr)
        return 2

    base_url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"
    t0 = time.time()
    plan = build_plan(base_url)
    bits = derive_bits(plan)
    total = sum(t.nbytes for t in plan)
    metadata = {
        "format": "mlx",
        "source": args.repo,
        "revision": args.revision,
        "norms": "folded",
        "gate_up_order": "gate_first",
    }
    hdr = header_bytes(plan, metadata)
    data_start = len(hdr)

    tmp_path = out_path + ".partial"
    fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, data_start + total)
        os.pwrite(fd, hdr, 0)
        transfer(base_url, plan, fd, data_start, args.workers)

        # fold +1 into the zero-centered norm weights
        stats = []
        for t in plan:
            if not t.name.endswith(CENTERED_SUFFIXES):
                continue
            if t.dtype != "BF16":
                raise RuntimeError(f"{t.name}: expected BF16 norm, got {t.dtype}")
            raw = os.pread(fd, t.nbytes, data_start + t.offset)
            before = bf16_to_f32(raw)
            folded = fold_one(raw)
            os.pwrite(fd, folded, data_start + t.offset)
            after = bf16_to_f32(folded)
            stats.append((t.name, before, after))
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.close(fd)
    os.replace(tmp_path, out_path)
    atomic_write_json(
        json_path,
        {
            "source": args.repo,
            "norms": "folded",
            "gate_up_order": "gate_first",
            "group_size": GROUP_SIZE,
            "bits": bits,
        },
    )

    # ------------------------------------------------------------------ self-checks
    print(f"\nwrote {out_path}\nwrote {json_path}")
    print(f"tensors: {len(plan)}   data bytes: {total} ({total / 1e9:.4f} GB)   file bytes: {os.path.getsize(out_path)}")
    print(f"elapsed: {time.time() - t0:.1f} s")
    names = {t.name for t in plan}
    quantized = [t.name for t in plan if t.dtype == "U32"]
    unfolded = [t.name for t in plan if t.dtype == "BF16" and t.name.endswith(".weight")
                and not t.name.endswith(CENTERED_SUFFIXES)]
    ok = all(n[:-7] + ".scales" in names and n[:-7] + ".biases" in names for n in quantized)
    print(f"quantized weights: {len(quantized)}; every .weight has .scales/.biases: {ok}; "
          f"un-quantized BF16 .weight that are not folded norms: {unfolded or 'none'}")
    mis = [t.name for t in plan if t.offset % (4 if t.dtype == "U32" else 2)]
    print(f"tensor offsets 8-byte aligned: {all(t.offset % 8 == 0 for t in plan)}; "
          f"U32/BF16 natural alignment violations: {mis or 'none'}")
    bit_hist: dict[int, int] = {}
    for b in bits.values():
        bit_hist[b] = bit_hist.get(b, 0) + 1
    print(f"bits histogram (group {GROUP_SIZE}): {dict(sorted(bit_hist.items()))}")
    print("8-bit bases:", sorted(k for k, v in bits.items() if v == 8))
    print(f"folded norms ({len(stats)}), raw -> folded  mean [min, max]:")
    for name, before, after in stats:
        print(f"  {name:70s} {before.mean():8.4f} -> {after.mean():8.4f}   "
              f"[{after.min():.4f}, {after.max():.4f}]  n={after.size}")
    if not ok or unfolded:
        print("SELF-CHECK FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
