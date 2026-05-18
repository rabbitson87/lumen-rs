#!/usr/bin/env python3
"""MLX runner — exposes both:

1. `MlxRunner` class: in-process API used by the PyO3 path. Each method
   returns a Python dict that PyO3 unpacks on the Rust side.

2. JSON-RPC subprocess loop (`if __name__ == '__main__'`): kept as fallback
   for environments where embedding Python isn't viable (no libpython linked,
   debugging, etc.). Set `LUMEN_MLX_SUBPROCESS=1` on the Rust side to use
   this path instead of PyO3.

Both paths share the same `MlxRunner` implementation — the only difference is
how arguments arrive (PyO3 method call vs JSON line on stdin).
"""

import json
import os
import sys
import time
import traceback


# ─────────────────────────────────────────────────────────────────────────────
# Embedded Python boot fix (PyO3 path)
#
# When this module is imported by a Rust binary that embeds CPython via PyO3,
# `sys.executable` points at the *Rust binary path*, not at the venv's Python.
# Any code that calls `multiprocessing.Process(method='spawn')` or uses
# `concurrent.futures.ProcessPoolExecutor` will then re-launch the *Rust
# binary* as a "Python child", which re-runs `main()` — including
# `MlxBackend::load`, which triggers another `mlx_lm.load()`. The result is
# repeated model loads + many extra processes (5×, 19 GB each → OOM).
#
# Concrete trigger: `huggingface_hub.snapshot_download` uses
# `concurrent.futures.ThreadPoolExecutor` by default but switches to a
# `ProcessPoolExecutor` for some operations. mlx-lm's loader can also go
# through process pools depending on safetensors plumbing.
#
# Fix: rewire `sys.executable` (and `multiprocessing.set_executable`) to the
# real venv Python before any code that might fork/spawn. Read the venv python
# path from `LUMEN_MLX_PYTHON` (set by the Rust side via .cargo/config.toml
# `PYO3_PYTHON`).
# ─────────────────────────────────────────────────────────────────────────────


def _fix_embedded_python_executable() -> None:
    venv_python = os.environ.get("LUMEN_MLX_PYTHON") or os.environ.get("PYO3_PYTHON")
    if venv_python and os.path.exists(venv_python):
        sys.executable = venv_python
        try:
            import multiprocessing
            multiprocessing.set_executable(venv_python)
        except Exception:
            pass


_fix_embedded_python_executable()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.6 — API shim for our locally-built patched MLX
#
# Our local MLX commit (08d6b01, ~0.30.6) predates upstream
# `mx.new_thread_local_stream` (used by mlx-lm 0.31.x's generate.py:226 at
# module import time). When this script imports our patched-MLX-using venv,
# the shim falls back to `new_stream` so mlx-lm loads. Keeps the lumen
# instrumentation accessible while preserving mlx-lm compatibility.
# ─────────────────────────────────────────────────────────────────────────────
def _install_mlx_compat_shim() -> None:
    try:
        import mlx.core as _mx
    except Exception:
        return
    if not hasattr(_mx, "new_thread_local_stream") and hasattr(_mx, "new_stream"):
        _mx.new_thread_local_stream = _mx.new_stream
        sys.stderr.write(
            "[mlx_runner] shim: new_thread_local_stream -> new_stream "
            "(local MLX predates upstream API)\n"
        )
        sys.stderr.flush()


_install_mlx_compat_shim()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.6 — Direct ctypes bridge to lumen-rs MLX instrumentation
#
# Our locally-built MLX exposes a handful of `extern "C"` entry points
# (mlx/transforms.cpp) so Python can read the per-primitive `gpu::eval`
# encode counters + dynamic primitive-name histogram WITHOUT going
# through MLX's pybind layer. This lets us compare op-by-op what mlx-lm
# emits vs what our native Rust path emits — both running through the
# same instrumented MLX binary.
#
# Symbols (file-scope, visibility=default to survive macOS dead-strip):
#   mlx_eval_gpu_calls_get()       -> u64
#   mlx_eval_gpu_ns_get()          -> u64
#   mlx_eval_gpu_stats_reset()     -> void
#   mlx_prim_hist_dyn_dump_buf(char*, int) -> int  (bytes written or -1)
#   mlx_prim_hist_dyn_reset_buf()  -> void
#
# Gracefully no-op when running against a stock (non-lumen) MLX wheel.
# ─────────────────────────────────────────────────────────────────────────────
_lumen_mlx_lib = None


def _load_lumen_counter_bindings():
    global _lumen_mlx_lib
    if _lumen_mlx_lib is not None:
        return _lumen_mlx_lib
    import ctypes
    import os
    try:
        import mlx.core as _mx
        # Counter atomics live in libmlx.dylib (separate from core.so).
        # Resolve `mlx/lib/libmlx.dylib` next to the core module.
        core_so = _mx.__file__
        lib_path = os.path.normpath(
            os.path.join(os.path.dirname(core_so), "lib", "libmlx.dylib")
        )
        if not os.path.exists(lib_path):
            sys.stderr.write(
                f"[mlx_runner] libmlx.dylib not found at {lib_path}; "
                "lumen counters unavailable\n"
            )
            return None
        lib = ctypes.CDLL(lib_path)
        lib.mlx_eval_gpu_calls_get.restype = ctypes.c_uint64
        lib.mlx_eval_gpu_calls_get.argtypes = []
        lib.mlx_eval_gpu_ns_get.restype = ctypes.c_uint64
        lib.mlx_eval_gpu_ns_get.argtypes = []
        lib.mlx_eval_gpu_stats_reset.restype = None
        lib.mlx_eval_gpu_stats_reset.argtypes = []
        lib.mlx_prim_hist_dyn_dump_buf.restype = ctypes.c_int
        lib.mlx_prim_hist_dyn_dump_buf.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.mlx_prim_hist_dyn_reset_buf.restype = None
        lib.mlx_prim_hist_dyn_reset_buf.argtypes = []
        # SDPA per-stage timing (mlx/fast.cpp namespace sdpa_timing).
        # `mlx_dump_sdpa_timing()` prints the breakdown to stderr;
        # `mlx_reset_sdpa_timing()` zeroes the counters between
        # warmup and the timed phase.
        lib.mlx_dump_sdpa_timing.restype = None
        lib.mlx_dump_sdpa_timing.argtypes = []
        lib.mlx_reset_sdpa_timing.restype = None
        lib.mlx_reset_sdpa_timing.argtypes = []
        _lumen_mlx_lib = lib
        return lib
    except Exception as e:
        sys.stderr.write(
            f"[mlx_runner] lumen counter bindings unavailable ({type(e).__name__}: {e})\n"
        )
        sys.stderr.flush()
        _lumen_mlx_lib = False
        return None


def reset_lumen_op_stats():
    """Reset eval_gpu + prim_hist counters in the Python MLX library."""
    lib = _load_lumen_counter_bindings()
    if not lib:
        return
    lib.mlx_eval_gpu_stats_reset()
    lib.mlx_prim_hist_dyn_reset_buf()


def reset_lumen_sdpa_timing():
    """Reset SDPA per-stage timing counters in mlx/fast.cpp."""
    lib = _load_lumen_counter_bindings()
    if not lib:
        return
    lib.mlx_reset_sdpa_timing()


def dump_lumen_sdpa_timing():
    """Print the SDPA per-stage breakdown (validation, astype, input_prep,
    fallback_check, primitive_ctor, fallback_path, TOTAL) to stderr."""
    lib = _load_lumen_counter_bindings()
    if not lib:
        return
    lib.mlx_dump_sdpa_timing()


def dump_lumen_op_stats(label: str = "pyo3", decode_steps: int = 0):
    """Print the eval_gpu encode totals + top primitive histogram entries
    to stderr. Mirrors the Rust-side dump in bench_gemma4_native_e2e so
    output is grep-comparable between paths."""
    import ctypes
    lib = _load_lumen_counter_bindings()
    if not lib:
        sys.stderr.write(
            "[op-stats] lumen counter bindings unavailable; cannot dump\n"
        )
        return
    calls = int(lib.mlx_eval_gpu_calls_get())
    ns_total = int(lib.mlx_eval_gpu_ns_get())
    ms = ns_total / 1e6
    per_call_us = (ns_total / 1000.0 / calls) if calls > 0 else 0.0
    per_step = (calls / decode_steps) if decode_steps > 0 else 0.0
    sys.stderr.write(
        f"[{label}-eval-gpu] calls={calls}  ns_total={ns_total}  ms={ms:.1f}  "
        f"per_call_us={per_call_us:.2f}  approx_per_step={per_step:.1f}\n"
    )
    # Primitive histogram dump (64 KiB staging buffer).
    buf = ctypes.create_string_buffer(65_536)
    n = lib.mlx_prim_hist_dyn_dump_buf(buf, len(buf))
    if n < 0:
        sys.stderr.write(
            f"[{label}-prim-hist] dump truncated (>64 KiB), see C++ counter\n"
        )
    else:
        text = buf.value.decode("utf-8", errors="replace")
        rows = []
        for line in text.split("\n"):
            if not line or "=" not in line:
                continue
            name, count = line.split("=", 1)
            try:
                rows.append((name, int(count)))
            except ValueError:
                continue
        rows.sort(key=lambda x: -x[1])
        total = sum(c for _, c in rows) or 1
        sys.stderr.write(
            f"[{label}-prim-hist] dynamic (sorted by count):\n"
        )
        for name, count in rows[:20]:
            pct = 100.0 * count / total
            sys.stderr.write(
                f"  {name:<40} {count:>8}  ({pct:.1f}%)\n"
            )
        if len(rows) > 20:
            sys.stderr.write(
                f"  ... {len(rows) - 20} more primitive types\n"
            )
    sys.stderr.flush()


# ─────────────────────────────────────────────────────────────────────────────
# DFlash speculative decode — env config + compatibility validation
#
# Phase D1.1: env-gated draft config load + startup compat checks. Default-off:
# when LUMEN_MLX_SPEC != "dflash", none of this code runs. The actual block-
# draft / target-verify / accept-reject loop is deferred to D1.2+. This pass
# only:
#   1. Detects DFlash mode env vars.
#   2. Locates and parses the draft model's config.json (no weight load).
#   3. Validates target↔draft compatibility (vocab, layer ids, mask, block).
#   4. Stores result on the runner so later phases can consume.
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_DFLASH_DRAFT = "z-lab/Qwen3.6-35B-A3B-DFlash"


def _read_dflash_env() -> "dict | None":
    """Return DFlash config dict when LUMEN_MLX_SPEC=dflash, else None.

    Recognized env vars:
      LUMEN_MLX_SPEC=dflash              — enable
      LUMEN_MLX_DFLASH_MODEL=<repo|path> — draft model id (default
                                              z-lab/Qwen3.6-35B-A3B-DFlash)
      LUMEN_MLX_DFLASH_BLOCK_SIZE=<int>  — override draft's declared
                                              block_size (rare; for tuning)
    """
    if os.environ.get("LUMEN_MLX_SPEC", "").strip() != "dflash":
        return None
    draft = os.environ.get("LUMEN_MLX_DFLASH_MODEL", _DEFAULT_DFLASH_DRAFT).strip()
    block_override = os.environ.get("LUMEN_MLX_DFLASH_BLOCK_SIZE")
    block_override_val = None
    if block_override:
        try:
            block_override_val = int(block_override)
            if block_override_val <= 0:
                raise ValueError
        except ValueError:
            sys.stderr.write(
                f"[mlx_runner] invalid LUMEN_MLX_DFLASH_BLOCK_SIZE={block_override!r}, ignoring\n"
            )
            block_override_val = None
    return {"draft_model_id": draft, "block_size_override": block_override_val}


def _dflash_locate_config(draft_model_id: str) -> str:
    """Resolve the draft model's local config.json path. Tries:
      1. `draft_model_id` as a literal path.
      2. HF Hub snapshot dir (already downloaded, offline-friendly).

    Raises with a clear message when neither resolves — the user must run
    `hf download <draft>` first, or set LUMEN_MLX_DFLASH_MODEL to a local
    path."""
    direct = os.path.join(draft_model_id, "config.json")
    if os.path.exists(direct):
        return direct
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            f"DFlash draft '{draft_model_id}' is not a local path and "
            f"huggingface_hub is unavailable: {e}"
        )
    try:
        snap = snapshot_download(
            repo_id=draft_model_id,
            allow_patterns=["config.json", "*.py"],
            local_files_only=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"DFlash draft '{draft_model_id}' not found locally and "
            f"local_files_only fetch failed: {e}. Run `hf download "
            f"{draft_model_id}` first."
        )
    cfg = os.path.join(snap, "config.json")
    if not os.path.exists(cfg):
        raise RuntimeError(f"config.json missing under {snap}")
    return cfg


def _dflash_parse_draft_config(config_path: str) -> dict:
    """Read and validate required DFlash draft config fields. Returns a
    normalized dict; raises on missing/malformed required fields."""
    with open(config_path, "r") as f:
        raw = json.load(f)
    required = ("vocab_size", "hidden_size", "num_hidden_layers", "block_size")
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        raise RuntimeError(
            f"DFlash draft config missing required fields {missing} in {config_path}"
        )
    dflash_block = raw.get("dflash_config")
    if not isinstance(dflash_block, dict):
        raise RuntimeError(
            f"DFlash draft config missing `dflash_config` block in {config_path}"
        )
    target_layer_ids = dflash_block.get("target_layer_ids")
    mask_token_id = dflash_block.get("mask_token_id")
    if not isinstance(target_layer_ids, list) or not all(isinstance(x, int) for x in target_layer_ids):
        raise RuntimeError(
            f"DFlash draft `target_layer_ids` must be list[int]; got {target_layer_ids!r}"
        )
    if not isinstance(mask_token_id, int):
        raise RuntimeError(
            f"DFlash draft `mask_token_id` must be int; got {mask_token_id!r}"
        )
    return {
        "config_path": config_path,
        "vocab_size": int(raw["vocab_size"]),
        "hidden_size": int(raw["hidden_size"]),
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "block_size": int(raw["block_size"]),
        "target_layer_ids": list(target_layer_ids),
        "mask_token_id": int(mask_token_id),
        "num_target_layers_declared": raw.get("num_target_layers"),
        "head_dim": raw.get("head_dim"),
        "num_attention_heads": raw.get("num_attention_heads"),
        "num_key_value_heads": raw.get("num_key_value_heads"),
        "model_type": raw.get("model_type"),
        "raw": raw,
    }


def _dflash_compat_check(
    target_vocab: int,
    target_num_layers: int,
    target_hidden_size: "int | None",
    draft_cfg: dict,
) -> tuple:
    """Run startup compatibility checks. Returns (errors: list[str],
    warnings: list[str]). Errors are fail-fast; warnings only logged."""
    errors: list[str] = []
    warnings: list[str] = []

    if draft_cfg["vocab_size"] != target_vocab:
        errors.append(
            f"vocab mismatch: target={target_vocab} draft={draft_cfg['vocab_size']}"
        )
    if not (0 <= draft_cfg["mask_token_id"] < draft_cfg["vocab_size"]):
        errors.append(
            f"mask_token_id={draft_cfg['mask_token_id']} out of vocab range "
            f"[0, {draft_cfg['vocab_size']})"
        )
    if not draft_cfg["target_layer_ids"]:
        errors.append("target_layer_ids is empty")
    bad_layers = [i for i in draft_cfg["target_layer_ids"] if not (0 <= i < target_num_layers)]
    if bad_layers:
        errors.append(
            f"target_layer_ids out of range: {bad_layers} (target has "
            f"{target_num_layers} layers)"
        )
    declared = draft_cfg.get("num_target_layers_declared")
    if isinstance(declared, int) and declared != target_num_layers:
        warnings.append(
            f"draft was trained against num_target_layers={declared}, current "
            f"target has {target_num_layers}; usually ok if architecture matches"
        )
    if draft_cfg["block_size"] <= 0 or draft_cfg["block_size"] > 256:
        errors.append(f"block_size={draft_cfg['block_size']} out of sane range (0, 256]")
    if target_hidden_size is not None and draft_cfg["hidden_size"] != target_hidden_size:
        errors.append(
            f"hidden_size mismatch: target={target_hidden_size} "
            f"draft={draft_cfg['hidden_size']} — DFlash cross-attention needs match"
        )
    return errors, warnings


# ─────────────────────────────────────────────────────────────────────────────
# DFlash D1.2 / D1.3: target hidden state capture + draft model
#
# The hooking primitive (`_LayerHook` proxy + `model._hidden_states` list) and
# the DFlash draft model itself live in `dflash_runtime`, vendored from the
# upstream `z-lab/dflash` MLX reference. We keep DFlash-specific code there
# so the generic `MlxRunner` stays small.
# ─────────────────────────────────────────────────────────────────────────────


def _read_kv_quant_env() -> tuple:
    """Parse MLX KV quantization env vars into (kv_bits, group_size, start).

    Returns (None, _, _) when MLX_KV_BITS unset — quantization disabled.
    Hybrid models (e.g., Qwen3.5 with ArraysCache for SSM/linear-attn layers)
    are safe: `maybe_quantize_kv_cache` skips caches without `to_quantized`.
    """
    bits_raw = os.environ.get("MLX_KV_BITS")
    if not bits_raw:
        return (None, 64, 0)
    try:
        bits = int(bits_raw)
    except ValueError:
        sys.stderr.write(f"[mlx_runner] invalid MLX_KV_BITS={bits_raw!r}, ignoring\n")
        return (None, 64, 0)
    if bits not in (2, 4, 8):
        sys.stderr.write(f"[mlx_runner] MLX_KV_BITS={bits} not in {{2,4,8}}, ignoring\n")
        return (None, 64, 0)
    try:
        group = int(os.environ.get("MLX_KV_GROUP_SIZE", "64"))
    except ValueError:
        group = 64
    try:
        start = int(os.environ.get("MLX_KV_QUANT_START", "0"))
    except ValueError:
        start = 0
    return (bits, group, start)


def _snapshot_cache(cache: list, deep: bool = False) -> dict:
    """Capture per-layer cache state.

    Two modes via `deep`:
      - **Shallow** (`deep=False`, default) — same-seq rollback (Track A2).
        Stores refs / offset only. Cheap (sub-ms). Restore is in-place mutation
        of the SAME seq's cache.
      - **Deep** (`deep=True`) — fork-to-new-seq (Track A1 prefix caching).
        Materializes independent mx.array allocations for state buffers so
        the snapshot can seed a *different* seq without aliasing the source.

    Returns `{layer_idx: (kind, payload)}`. Kinds:
      - 'arrays'      : list copy of c.cache (shallow refs)
      - 'arrays_deep' : list of fresh mx.array clones of c.cache
      - 'kv'          : c.offset (int) — restore via trim
      - 'kv_deep'     : (mx.array(keys), mx.array(values), offset)
      - 'quant_kv'    : c.offset — shallow only; deep returns 'unsupported'
      - 'rotating'    : (c.offset, c._idx) — shallow only
      - 'unsupported' : caller should reject the operation

    Deep mode forces `mx.eval` on each cloned array to materialize lazy
    graphs. Without this the clones would be views over the source's lazy
    chain and a subsequent mutation on the source would mutate the clone.
    """
    from mlx_lm.models import cache as mlx_cache

    snap = {}
    for i, c in enumerate(cache):
        if isinstance(c, mlx_cache.ArraysCache):
            if deep:
                import mlx.core as mx
                arrays = [mx.array(a) for a in c.cache]
                if arrays:
                    mx.eval(*arrays)
                snap[i] = ("arrays_deep", arrays)
            else:
                snap[i] = ("arrays", list(c.cache))
        elif isinstance(c, mlx_cache.QuantizedKVCache):
            if deep:
                # v1: quantized KV fork unsupported. Caller must disable KV
                # quant or take prefix snapshot before quant kicks in.
                snap[i] = ("unsupported", "quant_kv_deep_not_implemented")
            else:
                snap[i] = ("quant_kv", c.offset)
        elif isinstance(c, mlx_cache.RotatingKVCache):
            if deep:
                snap[i] = ("unsupported", "rotating_kv_deep_not_implemented")
            else:
                snap[i] = ("rotating", (c.offset, c._idx))
        elif isinstance(c, mlx_cache.KVCache):
            if deep:
                import mlx.core as mx
                kk = mx.array(c.keys) if c.keys is not None else None
                vv = mx.array(c.values) if c.values is not None else None
                if kk is not None:
                    mx.eval(kk, vv)
                snap[i] = ("kv_deep", (kk, vv, c.offset))
            else:
                snap[i] = ("kv", c.offset)
        else:
            snap[i] = ("unsupported", None)
    return snap


def _restore_cache(cache: list, snap: dict) -> None:
    """Apply `snap` into `cache`. Handles both same-seq rollback (shallow
    kinds: in-place trim/refswap) and fresh-cache install (deep kinds:
    fresh clones + attribute assignment). For deep kinds, an additional
    `mx.array(...)` clone happens at install time so that multiple forks
    from the same master snapshot remain mutually independent."""
    for i, c in enumerate(cache):
        if i not in snap:
            continue
        kind, payload = snap[i]
        if kind == "arrays":
            c.cache = list(payload)
        elif kind == "arrays_deep":
            import mlx.core as mx
            new_cache = [mx.array(a) for a in payload]
            if new_cache:
                mx.eval(*new_cache)
            c.cache = new_cache
        elif kind == "kv":
            target = payload
            current = c.offset
            if current > target:
                c.trim(current - target)
        elif kind == "kv_deep":
            import mlx.core as mx
            kk, vv, off = payload
            c.keys = mx.array(kk) if kk is not None else None
            c.values = mx.array(vv) if vv is not None else None
            c.offset = off
            if c.keys is not None:
                mx.eval(c.keys, c.values)
        elif kind == "quant_kv":
            target = payload
            current = c.offset
            if current > target:
                c.trim(current - target)
        elif kind == "rotating":
            target_off, _target_idx = payload
            current = c.offset
            if current > target_off:
                c.trim(current - target_off)
        # 'unsupported': no-op


class MlxRunner:
    """In-process MLX backend. Holds the loaded model + per-seq prompt caches."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        # seq_id (int) -> (prompt_cache, position (int))
        self.states = {}
        self.kv_bits, self.kv_group_size, self.kv_quant_start = _read_kv_quant_env()
        self._maybe_quantize = None
        # snapshot_id (int) -> (snap_dict, position) for spec-decode rollback.
        self._snapshots = {}
        self._next_snapshot_id = 1
        # DFlash spec-decode runtime state. Populated by _dflash_init() inside
        # load() when LUMEN_MLX_SPEC=dflash is set, else stays None.
        # When enabled, the loaded MLX target carries its captured hidden
        # states list at `self.model._hidden_states` (installed by
        # `dflash_runtime.patch_model`), and `self._dflash_draft` holds the
        # loaded DFlash draft model + its KV cache.
        self.dflash = None
        self._dflash_draft = None
        # Per-seq DFlash auxiliary state: {seq_id: {draft_cache, target_hidden}}.
        # Lives parallel to `self.states` so baseline prefill/decode_step paths
        # stay untouched. `target_hidden` is the cross-attn context fed to the
        # draft on the next block: full-prompt hidden after dflash_prefill,
        # then trimmed to (accepted+1) positions after each block.
        self.dflash_states = {}
        # GatedDeltaNet rollback capture for hybrid Qwen targets where some
        # cache layers are not trimmable. Lazily instantiated in the first
        # `dflash_block_step` that observes a non-trimmable cache. Single
        # instance class-monkeypatches GDN.__call__ globally — single-seq
        # invariant of MlxRunner makes this safe.
        self._gdn_capture = None

    def load(self, model_id: str) -> dict:
        from mlx_lm import load
        from mlx_lm.generate import maybe_quantize_kv_cache

        self._maybe_quantize = maybe_quantize_kv_cache

        sys.stderr.write(f"[mlx_runner] loading {model_id}...\n")
        sys.stderr.flush()
        t0 = time.time()
        self.model, self.tokenizer = load(model_id)
        load_ms = (time.time() - t0) * 1000.0
        sys.stderr.write(f"[mlx_runner] loaded in {load_ms:.0f}ms\n")
        if self.kv_bits is not None:
            sys.stderr.write(
                f"[mlx_runner] KV cache quantization ENABLED: "
                f"bits={self.kv_bits} group={self.kv_group_size} "
                f"start_offset={self.kv_quant_start}\n"
            )
        sys.stderr.flush()
        eos_ids = []
        for tok_str in ["<|im_end|>", "<|endoftext|>"]:
            tid = self.tokenizer.encode(tok_str, add_special_tokens=False)
            if len(tid) == 1:
                eos_ids.append(int(tid[0]))
        vocab = int(self.tokenizer.vocab_size) if hasattr(self.tokenizer, "vocab_size") else 0
        # DFlash compat must check against the model's embedding-matrix vocab,
        # not the tokenizer's reported vocab — Qwen pads its embedding rows
        # past the tokenizer max (e.g. 248320 vs 248044) and the mask token id
        # often lives in that padded range.
        model_vocab = self._target_vocab_size() or vocab
        dflash_status = self._dflash_init(model_vocab) if _read_dflash_env() is not None else None
        return {
            "load_ms": load_ms,
            "eos_tokens": eos_ids,
            "vocab_size": vocab,
            "dflash": dflash_status,
        }

    def _walk_attr(self, paths: tuple) -> "object | None":
        """Try multiple attribute paths on self.model; return first hit."""
        for path in paths:
            obj = self.model
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                return obj
        return None

    def _target_num_layers(self) -> int:
        """Best-effort introspection of the loaded MLX target's layer count.
        Falls back to 0 when the target shape is unfamiliar — compat check
        will then mark target_layer_ids as out-of-range and fail-fast.

        Layouts seen in the wild:
          - qwen3_5_moe (multimodal-wrapped): model.language_model.layers
            (via @property) + model.language_model.args.num_hidden_layers
          - qwen3_moe (unwrapped):           model.model.layers +
            model.model.args.num_hidden_layers
          - other:                           model.layers / model.args.*
        """
        layers = self._walk_attr((
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        ))
        if layers is not None:
            try:
                return int(len(layers))
            except TypeError:
                pass
        n = self._walk_attr((
            ("language_model", "args", "num_hidden_layers"),
            ("args", "num_hidden_layers"),
            ("model", "args", "num_hidden_layers"),
        ))
        return int(n) if isinstance(n, int) else 0

    def _target_hidden_size(self) -> "int | None":
        h = self._walk_attr((
            ("language_model", "args", "hidden_size"),
            ("args", "hidden_size"),
            ("model", "args", "hidden_size"),
        ))
        return int(h) if isinstance(h, int) else None

    def _target_vocab_size(self) -> "int | None":
        """Introspect the loaded MLX target's embedding-matrix vocab size.
        Used by DFlash compat (mask token may live in tokenizer-vs-model
        padded range)."""
        v = self._walk_attr((
            ("language_model", "args", "vocab_size"),
            ("args", "vocab_size"),
            ("model", "args", "vocab_size"),
        ))
        return int(v) if isinstance(v, int) else None

    def _dflash_init(self, target_vocab: int) -> dict:
        """Phase D1.1: load DFlash draft config + run startup compatibility
        checks. Sets `self.dflash` on success. On compat error, raises so the
        caller (Rust engine) sees a clean fail-fast — DFlash never silently
        downgrades to vanilla decode.

        Returns a status dict suitable for logging / Rust-side ingestion. Does
        NOT load draft weights or instantiate any nn.Module — that is D1.2+."""
        env = _read_dflash_env()
        assert env is not None, "_dflash_init called with DFlash env disabled"
        draft_id = env["draft_model_id"]
        config_path = _dflash_locate_config(draft_id)
        draft_cfg = _dflash_parse_draft_config(config_path)

        if env["block_size_override"] is not None:
            sys.stderr.write(
                f"[mlx_runner] dflash: overriding block_size "
                f"{draft_cfg['block_size']} -> {env['block_size_override']}\n"
            )
            draft_cfg["block_size"] = env["block_size_override"]

        target_num_layers = self._target_num_layers()
        target_hidden_size = self._target_hidden_size()
        errors, warnings = _dflash_compat_check(
            target_vocab=target_vocab,
            target_num_layers=target_num_layers,
            target_hidden_size=target_hidden_size,
            draft_cfg=draft_cfg,
        )

        for w in warnings:
            sys.stderr.write(f"[mlx_runner] dflash WARN: {w}\n")
        if errors:
            for e in errors:
                sys.stderr.write(f"[mlx_runner] dflash ERROR: {e}\n")
            sys.stderr.flush()
            raise RuntimeError(
                f"DFlash compatibility check failed ({len(errors)} error(s)); "
                f"first: {errors[0]}"
            )

        sys.stderr.write(
            f"[mlx_runner] dflash compat ok: draft={draft_id} "
            f"vocab={draft_cfg['vocab_size']} block={draft_cfg['block_size']} "
            f"target_layers={draft_cfg['target_layer_ids']} "
            f"mask_token={draft_cfg['mask_token_id']} "
            f"target_num_layers={target_num_layers} "
            f"target_hidden_size={target_hidden_size}\n"
        )
        sys.stderr.flush()

        status = {
            "enabled": True,
            "draft_model_id": draft_id,
            "config_path": config_path,
            "block_size": draft_cfg["block_size"],
            "mask_token_id": draft_cfg["mask_token_id"],
            "target_layer_ids": draft_cfg["target_layer_ids"],
            "draft_hidden_size": draft_cfg["hidden_size"],
            "draft_num_hidden_layers": draft_cfg["num_hidden_layers"],
            "target_num_layers": target_num_layers,
            "target_hidden_size": target_hidden_size,
            "warnings": warnings,
            "weights_loaded": False,
        }
        self.dflash = status

        # D1.2 — install per-target-layer capture hooks (proxy pattern from
        # upstream). Storage is a list at `self.model._hidden_states`,
        # enumeration-indexed so concatenation along the channel axis matches
        # the draft's `fc` weight layout.
        import dflash_runtime as _dr
        n_wrapped = _dr.patch_model(self.model, draft_cfg["target_layer_ids"])
        sys.stderr.write(
            f"[mlx_runner] dflash capture hooks installed on {n_wrapped} "
            f"target layer(s) at {draft_cfg['target_layer_ids']}\n"
        )
        sys.stderr.flush()
        status["capture_layers_wrapped"] = n_wrapped

        # D1.3.0 — load + bind the DFlash draft. `bind` borrows the target's
        # `embed_tokens` and `lm_head` so the draft shares the input/output
        # vocabulary projection without duplicating those (large) weights.
        # The draft KV cache is per-seq, allocated by `dflash_prefill`.
        sys.stderr.write(f"[mlx_runner] dflash loading draft {draft_id}...\n")
        sys.stderr.flush()
        t0 = time.time()
        self._dflash_draft = _dr.load_draft(draft_id)
        self._dflash_draft.bind(self.model)
        load_ms = (time.time() - t0) * 1000.0
        from mlx.utils import tree_flatten
        flat = tree_flatten(self._dflash_draft.parameters())
        n_params = sum(arr.size for _k, arr in flat)
        sys.stderr.write(
            f"[mlx_runner] dflash draft loaded in {load_ms:.0f}ms — "
            f"params={n_params/1e6:.1f}M layers={self._dflash_draft.config.num_hidden_layers} "
            f"head_dim={self._dflash_draft.config.head_dim}\n"
        )
        sys.stderr.flush()
        status["weights_loaded"] = True
        status["draft_load_ms"] = load_ms
        status["draft_params"] = n_params

        return status

    def _target_layers_list(self):
        """Return the list of target's transformer layers, or None on
        unrecognized layout. Mirrors _target_num_layers introspection."""
        return self._walk_attr((
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        ))

    def _dflash_get_target_hidden(self):
        """Return the captured target hidden states concatenated along the
        channel axis as a single mx.array of shape
        `[B, T, n_layers * hidden_size]`. Returns None when DFlash is disabled
        or no forward has populated the capture slots."""
        if self.dflash is None:
            return None
        import dflash_runtime as _dr
        return _dr.get_target_hidden(self.model)

    def _dflash_clear_captures(self) -> None:
        """Reset the capture slots to None. Useful between sequences or
        before idle periods. Releases lazy-graph references the captured
        tensors hold."""
        states = getattr(self.model, "_hidden_states", None)
        if states is None:
            return
        for i in range(len(states)):
            states[i] = None

    def _dflash_capture_status(self) -> dict:
        """Diagnostics: per-slot captured shape + dtype, plus whether all
        target layers have been captured. Cheap — does not force eval."""
        states = getattr(self.model, "_hidden_states", None)
        target_layer_ids = (
            self.dflash["target_layer_ids"] if self.dflash is not None else []
        )
        per_layer = {}
        if states is not None:
            for i, t in enumerate(states):
                if t is None:
                    continue
                lid = target_layer_ids[i] if i < len(target_layer_ids) else i
                shape = tuple(t.shape) if hasattr(t, "shape") else None
                dtype = str(getattr(t, "dtype", None))
                per_layer[lid] = {"shape": shape, "dtype": dtype}
        complete = (
            states is not None
            and bool(target_layer_ids)
            and all(s is not None for s in states)
        )
        return {
            "enabled": self.dflash is not None,
            "per_layer": per_layer,
            "complete": complete,
            "target_layer_ids": target_layer_ids,
            "draft_loaded": self._dflash_draft is not None,
        }

    def _make_cache(self):
        from mlx_lm.models import cache as mlx_cache
        return mlx_cache.make_prompt_cache(self.model)

    def _apply_kv_quant(self, cache: list) -> None:
        """Convert eligible per-layer caches to QuantizedKVCache once their
        offset crosses `kv_quant_start`. Safe to call after every step;
        layers without `to_quantized` (e.g., ArraysCache) are skipped."""
        if self.kv_bits is None or self._maybe_quantize is None:
            return
        self._maybe_quantize(
            cache, self.kv_quant_start, self.kv_group_size, self.kv_bits
        )

    def prefill(self, seq_id: int, tokens: list[int]) -> dict:
        if self.model is None:
            raise RuntimeError("model not loaded")
        if seq_id in self.states:
            raise RuntimeError(f"seq_id {seq_id} already exists")
        if not tokens:
            raise RuntimeError("empty tokens")
        import mlx.core as mx

        cache = self._make_cache()
        prompt = mx.array(tokens, dtype=mx.uint32)
        if len(tokens) > 1:
            prefix = prompt[:-1]
            logits = self.model(prefix[None], cache=cache)
            mx.eval(logits)
        last = prompt[-1:]
        logits = self.model(last[None], cache=cache)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        position = len(tokens)
        self._apply_kv_quant(cache)
        self.states[seq_id] = (cache, position)
        return {"next_token": next_tok, "position": position}

    def decode_step(self, seq_id: int, last_token: int, position: int) -> dict:
        if self.model is None:
            raise RuntimeError("model not loaded")
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        import mlx.core as mx
        import os, time

        # Stage timing for Native-vs-PyO3 root-cause analysis. Enabled by
        # LUMEN_PYO3_DECODE_STAGE_TIMING=1. Stores per-step (ns) tuple:
        #   (arr_ns, forward_ns, sync_ns, tail_ns)
        # where forward_ns is the lazy graph build (returns immediately for
        # mlx's lazy evaluation) and sync_ns is the argmax + .item() that
        # forces GPU sync. tail_ns is _apply_kv_quant + state update.
        stage_timing = os.environ.get("LUMEN_PYO3_DECODE_STAGE_TIMING") == "1"
        if stage_timing:
            t0 = time.perf_counter_ns()
        cache, pos = self.states[seq_id]
        arr = mx.array([[last_token]], dtype=mx.uint32)
        if stage_timing:
            t_arr = time.perf_counter_ns()
        logits = self.model(arr, cache=cache)
        if stage_timing:
            t_forward = time.perf_counter_ns()
        # Skip explicit mx.eval — argmax(...).item() forces sync.
        next_tok = int(mx.argmax(logits[0, -1]).item())
        if stage_timing:
            t_sync = time.perf_counter_ns()
        new_position = pos + 1
        self._apply_kv_quant(cache)
        self.states[seq_id] = (cache, new_position)
        if stage_timing:
            t_end = time.perf_counter_ns()
            if not hasattr(self, "_decode_stage_timings"):
                self._decode_stage_timings = []
            self._decode_stage_timings.append(
                (t_arr - t0, t_forward - t_arr, t_sync - t_forward, t_end - t_sync)
            )
        return {"next_token": next_tok, "position": new_position}

    def get_decode_stage_timings(self) -> list:
        """Drain and return per-step stage timings collected when
        LUMEN_PYO3_DECODE_STAGE_TIMING=1 was set during decode_step calls.
        Each entry: (arr_ns, forward_ns, sync_ns, tail_ns)."""
        if not hasattr(self, "_decode_stage_timings"):
            return []
        out = self._decode_stage_timings
        self._decode_stage_timings = []
        return out

    def extend(self, seq_id: int, tokens: list[int]) -> dict:
        """Feed `tokens` (length >= 1) into an existing cache without resetting.
        Used for prompt-cache reuse: the suffix of a new turn after the
        longest-common-prefix with prior session state. Returns the argmax of
        the *last* logits (next token after extension) and the new position.
        """
        if self.model is None:
            raise RuntimeError("model not loaded")
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        if not tokens:
            raise RuntimeError("empty tokens for extend")
        import mlx.core as mx

        cache, pos = self.states[seq_id]
        arr = mx.array(tokens, dtype=mx.uint32)
        if len(tokens) > 1:
            prefix = arr[:-1]
            logits = self.model(prefix[None], cache=cache)
            mx.eval(logits)
        last = arr[-1:]
        logits = self.model(last[None], cache=cache)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        new_position = pos + len(tokens)
        self._apply_kv_quant(cache)
        self.states[seq_id] = (cache, new_position)
        return {"next_token": next_tok, "position": new_position}

    def dflash_prefill(self, seq_id: int, tokens: list[int]) -> dict:
        """DFlash-mode prefill. Single-forward over the full prompt to capture
        the full-T target hidden states used as cross-attn context for the
        first block. Allocates a fresh per-seq draft KV cache.

        Diverges from the baseline `prefill` which does a 2-stage forward
        (`prefix[:-1]` + `last[-1:]`). Under that 2-stage split, the
        `_LayerHook` capture slots are overwritten by the second forward and
        only carry the last token's hiddens — DFlash needs the entire prompt's
        hidden trace as cross-attn K/V."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        if self.dflash is None:
            raise RuntimeError("dflash_prefill requires LUMEN_MLX_SPEC=dflash")
        if seq_id in self.states:
            raise RuntimeError(f"seq_id {seq_id} already exists")
        if not tokens:
            raise RuntimeError("empty tokens")
        import mlx.core as mx
        import dflash_runtime as _dr

        # Reset capture slots so any stale state from a prior forward (e.g.,
        # a probe call) doesn't leak into this seq's hidden trace.
        self._dflash_clear_captures()

        target_cache = self._make_cache()
        prompt = mx.array(tokens, dtype=mx.uint32)
        logits = self.model(prompt[None], cache=target_cache)
        target_hidden = _dr.get_target_hidden(self.model)
        if target_hidden is None:
            raise RuntimeError(
                "DFlash target hidden capture incomplete after prefill — "
                "verify patch_model installed hooks on every layer in "
                "target_layer_ids"
            )
        mx.eval(logits, target_hidden)
        next_tok = int(mx.argmax(logits[0, -1]).item())

        draft_cache = self._dflash_draft.make_cache()

        position = len(tokens)
        self._apply_kv_quant(target_cache)
        self.states[seq_id] = (target_cache, position)
        self.dflash_states[seq_id] = {
            "draft_cache": draft_cache,
            "target_hidden": target_hidden,
        }
        return {"next_token": next_tok, "position": position}

    def _dflash_ensure_gdn_capture(self, target_cache) -> "object | None":
        """Lazily instantiate GDNStateCapture when the target cache contains
        non-trimmable layers (hybrid Qwen with linear-attn). Returns None for
        fully-trimmable targets so `trim_prompt_cache` is sufficient.

        Single-instance because GDNStateCapture class-monkeypatches
        `GatedDeltaNet.__call__` globally; multiple concurrent instances would
        deadlock on `_GDN_PATCH_LOCK`. The single-seq invariant of MlxRunner
        keeps this safe."""
        from mlx_lm.models.cache import can_trim_prompt_cache

        if can_trim_prompt_cache(target_cache):
            return None
        if self._gdn_capture is None:
            import dflash_runtime as _dr
            sys.stderr.write(
                "[mlx_runner] dflash: target has non-trimmable cache layers "
                "(GDN/linear-attn); installing GDNStateCapture for rollback\n"
            )
            sys.stderr.flush()
            self._gdn_capture = _dr.GDNStateCapture()
        return self._gdn_capture

    def dflash_block_step(self, seq_id: int, last_token: int) -> dict:
        """DFlash speculative block step. One block = up to `block_size`
        candidate tokens drafted in parallel, then verified against the target
        in a single forward. Greedy accept-prefix.

        Mirrors upstream `stream_generate`'s inner loop, distilled to:
          1. draft sees `[last_token, MASK*(bs-1)]`, predicts a block.
          2. target verifies `[last_token, draft_tokens]` in one forward.
          3. Accept first-mismatch prefix; corrected target token replaces
             the rejected position.
          4. Roll back target cache by `bs - accepted - 1`; trim draft cache
             via the upstream offset formula. GDN-replay rollback for
             non-trimmable hybrid layers.
          5. Update stored target_hidden to the accepted prefix's slice for
             the next block's cross-attn ctx.

        Returns:
          accepted   — number of draft tokens matched (0..bs-1)
          new_tokens — list[int] of length accepted+1 (drafted prefix + 1
                       corrected target token)
          position   — new committed position
        """
        if self.model is None:
            raise RuntimeError("model not loaded")
        if self.dflash is None:
            raise RuntimeError("dflash_block_step requires LUMEN_MLX_SPEC=dflash")
        if seq_id not in self.states or seq_id not in self.dflash_states:
            raise RuntimeError(f"seq_id {seq_id} not initialized for dflash")
        import mlx.core as mx
        import dflash_runtime as _dr
        from mlx_lm.models.cache import trim_prompt_cache

        target_cache, position = self.states[seq_id]
        aux = self.dflash_states[seq_id]
        draft_cache = aux["draft_cache"]
        target_hidden = aux["target_hidden"]

        bs = int(self.dflash["block_size"])
        mask_id = int(self.dflash["mask_token_id"])
        if bs < 2:
            raise RuntimeError(f"DFlash block_size must be >= 2, got {bs}")

        capture = self._dflash_ensure_gdn_capture(target_cache)

        # ── Draft forward ─────────────────────────────────────────────────
        # Block input: [last, MASK*(bs-1)]. Draft predicts at every position;
        # we keep positions 1..bs-1 as the speculative tokens (position 0's
        # prediction is target's job — it sees `last` directly during verify).
        block = mx.array(
            [[last_token] + [mask_id] * (bs - 1)], dtype=mx.uint32
        )
        draft_logits = self._dflash_draft(block, target_hidden, draft_cache)

        # Trim draft cache (matches upstream's `prompt.size + n - 1` formula:
        # in our framing, that quantity equals `position` at this point).
        if self._dflash_draft.config.sliding_window_size is None:
            trim_n = draft_cache[0].offset - position
            if trim_n > 0:
                trim_prompt_cache(draft_cache, trim_n)

        draft_tokens = mx.argmax(draft_logits[:, 1 - bs:], axis=-1).astype(
            mx.uint32
        )

        # ── Target verify ─────────────────────────────────────────────────
        # Reset capture slots + GDN capture so the verify forward repopulates.
        self._dflash_clear_captures()
        if capture is not None:
            capture.clear()

        last_arr = mx.array([[last_token]], dtype=mx.uint32)
        verify_input = mx.concatenate([last_arr, draft_tokens], axis=1)
        target_logits = self.model(verify_input, cache=target_cache)
        new_target_hidden = _dr.get_target_hidden(self.model)
        if new_target_hidden is None:
            raise RuntimeError(
                "DFlash target hidden capture incomplete after verify forward"
            )
        target_tokens = mx.argmax(target_logits, axis=-1)
        mx.eval(draft_tokens, target_tokens, new_target_hidden)

        # ── Accept-prefix (greedy) ────────────────────────────────────────
        d_list = draft_tokens[0].tolist()
        t_list = target_tokens[0].tolist()
        # `accepted` is the number of consecutive matches starting at index 0.
        # Range [0, len(d_list)] = [0, bs-1]. When all match: accepted=bs-1,
        # we pick t_list[bs-1] as the (bs)-th committed token.
        accepted = next(
            (i for i in range(len(d_list)) if d_list[i] != t_list[i]),
            len(d_list),
        )
        new_tokens = d_list[:accepted] + [t_list[accepted]]

        # ── Rollback ──────────────────────────────────────────────────────
        # Target cache advanced by bs (full block). We commit accepted+1 of
        # those, so trim the rest.
        trim = bs - accepted - 1
        if trim > 0:
            if capture is None:
                trim_prompt_cache(target_cache, trim)
            else:
                capture.rollback(target_cache, accepted, trim)

        # Carry forward only the accepted prefix's hidden states for the
        # next block's cross-attn context.
        next_target_hidden = new_target_hidden[:, : accepted + 1, :]

        new_position = position + accepted + 1
        self._apply_kv_quant(target_cache)
        self.states[seq_id] = (target_cache, new_position)
        aux["target_hidden"] = next_target_hidden

        return {
            "accepted": int(accepted),
            "new_tokens": [int(t) for t in new_tokens],
            "position": new_position,
        }

    def remove_seq(self, seq_id: int) -> None:
        # Phase 1.6 — auto-dump lumen op stats on the last seq removal of
        # a bench. Gated by `LUMEN_EVAL_GPU_DUMP=1`, mirroring the env
        # the native bench uses so a single env-var flip dumps both paths.
        if os.environ.get("LUMEN_EVAL_GPU_DUMP", "0") == "1":
            # Approximate decode_steps via the per-step timing log size
            # (one entry per decode_step call when stage timing is on).
            n_steps = 0
            if hasattr(self, "_decode_stage_timings"):
                n_steps = len(self._decode_stage_timings)
            dump_lumen_op_stats(label="pyo3", decode_steps=n_steps)
        # Phase 1.6 — SDPA per-stage timing dump from the Python path,
        # using the same counter mechanism the Rust bench reads. Lets us
        # compare per-call SDPA cost between PyO3 and native paths against
        # the SAME instrumented MLX binary.
        if os.environ.get("LUMEN_SDPA_TIMING_DUMP", "0") == "1":
            dump_lumen_sdpa_timing()
        self.states.pop(seq_id, None)
        self.dflash_states.pop(seq_id, None)
        # Drop any snapshots tied to this seq id. Snapshots are per-cache, but
        # we don't track ownership — restore is only meaningful within the
        # same seq's lifecycle, so a subsequent prefill of the same id would
        # invalidate them anyway.

    def snapshot_state(self, seq_id: int) -> dict:
        """Capture the per-layer cache state for short-horizon spec-decode
        rollback. Returns `{snapshot_id: int}`. Caller must `restore_state`
        or `release_snapshot` to free the captured arrays."""
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        cache, pos = self.states[seq_id]
        snap = _snapshot_cache(cache)
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (snap, pos)
        return {"snapshot_id": sid}

    def restore_state(self, seq_id: int, snapshot_id: int) -> dict:
        """Restore the seq's cache to a previously captured snapshot.
        Snapshot is consumed (released) so a single snapshot is one-shot."""
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        if snapshot_id not in self._snapshots:
            raise RuntimeError(f"unknown snapshot_id {snapshot_id}")
        snap, pos = self._snapshots.pop(snapshot_id)
        cache, _ = self.states[seq_id]
        _restore_cache(cache, snap)
        self.states[seq_id] = (cache, pos)
        return {"position": pos}

    def release_snapshot(self, snapshot_id: int) -> None:
        self._snapshots.pop(snapshot_id, None)

    def snapshot_state_deep(self, seq_id: int) -> dict:
        """Like `snapshot_state` but materializes independent clones of the
        cache state. Suitable for seeding a different seq's cache via
        `fork_from_snapshot`. Master snapshot is reusable across many forks
        (release_snapshot to free).

        Returns `{snapshot_id: int, position: int}`. Raises if any layer's
        cache type can't be deep-cloned (quantized KV, rotating KV) — in that
        case caller should disable KV quant or fork before activation."""
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        cache, pos = self.states[seq_id]
        snap = _snapshot_cache(cache, deep=True)
        unsupported = [
            (i, payload) for i, (k, payload) in snap.items() if k == "unsupported"
        ]
        if unsupported:
            preview = unsupported[:3]
            raise RuntimeError(
                f"deep snapshot blocked by {len(unsupported)} layer(s); first: {preview}"
            )
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (snap, pos)
        return {"snapshot_id": sid, "position": pos}

    def fork_from_snapshot(self, snapshot_id: int, dst_seq_id: int) -> dict:
        """Create a fresh seq `dst_seq_id` whose cache is initialized by deep-
        cloning the snapshot's state. Snapshot is NOT consumed — multiple
        forks from the same master snapshot are independent (each install
        clones again at attribute-assignment time). Returns the destination
        seq's starting position."""
        if snapshot_id not in self._snapshots:
            raise RuntimeError(f"unknown snapshot_id {snapshot_id}")
        if dst_seq_id in self.states:
            raise RuntimeError(f"dst seq_id {dst_seq_id} already exists")
        snap, pos = self._snapshots[snapshot_id]
        new_cache = self._make_cache()
        _restore_cache(new_cache, snap)
        self.states[dst_seq_id] = (new_cache, pos)
        return {"position": pos}

    def forward_probe(self, seq_id: int, tokens: list[int]) -> dict:
        """Single batched forward of `tokens` (length K) at the seq's current
        cache state. Returns row-by-row argmax + max-abs-logit, advancing the
        seq's position by K. Used by Track A2 drift baseline + spec-decode
        verify-loop. Caller is responsible for `remove_seq` afterwards (or
        accepts the advanced state)."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        if seq_id not in self.states:
            raise RuntimeError(f"seq_id {seq_id} not initialized")
        if not tokens:
            raise RuntimeError("empty tokens for forward_probe")
        import mlx.core as mx

        cache, pos = self.states[seq_id]
        arr = mx.array(tokens, dtype=mx.uint32)[None]
        logits = self.model(arr, cache=cache)
        mx.eval(logits)
        K = arr.shape[1]
        argmaxes = mx.argmax(logits[0], axis=-1)
        maxabs = mx.max(mx.abs(logits[0]), axis=-1)
        mx.eval(argmaxes, maxabs)
        row_argmaxes = [int(argmaxes[i].item()) for i in range(K)]
        row_max_abs = [float(maxabs[i].item()) for i in range(K)]
        new_position = pos + K
        self._apply_kv_quant(cache)
        self.states[seq_id] = (cache, new_position)
        return {
            "row_argmaxes": row_argmaxes,
            "row_max_abs": row_max_abs,
            "position": new_position,
        }


# ────────────────────────────────────────────────────────────────────────────
# Subprocess JSON-RPC fallback (used by LUMEN_MLX_SUBPROCESS=1)
# ────────────────────────────────────────────────────────────────────────────

def _send(obj):
    sys.stdout.write(json.dumps(obj))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(msg):
    _send({"ok": False, "err": msg})


def _ok(extra=None):
    out = {"ok": True}
    if extra:
        out.update(extra)
    _send(out)


def _main_subprocess():
    runner = MlxRunner()
    sys.stderr.write("[mlx_runner] ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _err(f"bad json: {e}")
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "load":
                _ok(runner.load(req["model_id"]))
            elif cmd == "prefill":
                _ok(runner.prefill(int(req["seq_id"]), req["tokens"]))
            elif cmd == "decode_step":
                _ok(runner.decode_step(int(req["seq_id"]), int(req["last_token"]), int(req["position"])))
            elif cmd == "extend":
                _ok(runner.extend(int(req["seq_id"]), req["tokens"]))
            elif cmd == "forward_probe":
                _ok(runner.forward_probe(int(req["seq_id"]), req["tokens"]))
            elif cmd == "snapshot_state":
                _ok(runner.snapshot_state(int(req["seq_id"])))
            elif cmd == "restore_state":
                _ok(runner.restore_state(int(req["seq_id"]), int(req["snapshot_id"])))
            elif cmd == "release_snapshot":
                runner.release_snapshot(int(req["snapshot_id"]))
                _ok()
            elif cmd == "snapshot_state_deep":
                _ok(runner.snapshot_state_deep(int(req["seq_id"])))
            elif cmd == "fork_from_snapshot":
                _ok(runner.fork_from_snapshot(int(req["snapshot_id"]), int(req["dst_seq_id"])))
            elif cmd == "remove_seq":
                runner.remove_seq(int(req["seq_id"]))
                _ok()
            elif cmd == "dflash_prefill":
                _ok(runner.dflash_prefill(int(req["seq_id"]), req["tokens"]))
            elif cmd == "dflash_block_step":
                _ok(runner.dflash_block_step(int(req["seq_id"]), int(req["last_token"])))
            elif cmd == "shutdown":
                _ok({"bye": True})
                sys.exit(0)
            else:
                _err(f"unknown cmd: {cmd}")
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            sys.stderr.flush()
            _err(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    _main_subprocess()
