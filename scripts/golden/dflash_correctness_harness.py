"""DFlash D2 correctness harness.

Compares DFlash speculative decoding (`dflash_prefill` + `dflash_block_step`)
against baseline greedy decoding (`prefill` + `decode_step`) on the
Qwen3.6-35B-A3B-mxfp4 target / Qwen3.6-35B-A3B-DFlash draft pair.

Checks performed (greedy temp=0):
  1. Bit-identical token sequences across modes for N output tokens.
  2. Per-block cache position invariants: target/draft cache offset advances
     consistently with the position counter, equal to position post-rollback.
  3. EOS-inside-block handling: when EOS appears in `new_tokens`, caller
     truncates output and stops. Both modes terminate at the same EOS index.
  4. Per-token transcript logging for both paths.

Opt-in (requires both target and draft locally cached). Run with:

    LUMEN_MLX_DFLASH_MODEL=z-lab/Qwen3.6-35B-A3B-DFlash \\
    python3 scripts/dflash_correctness_harness.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(
    0,
    "/path/to/Documents/GitHub/lumen-rs/crates/lumen-mlx/python",
)

import mlx.core as mx
from mlx_runner import MlxRunner

MODEL_ID = "mlx-community/Qwen3.6-35B-A3B-mxfp4"
DRAFT_ID = "z-lab/Qwen3.6-35B-A3B-DFlash"

# Two prompts: open-ended completion (drives long output) + chat-style
# closer to EOS (drives early termination through the model's instruction-tuning).
PROMPTS = {
    "open": "The capital of France is",
    "chat": (
        "<|im_start|>user\nReply with exactly the word 'OK' and nothing else."
        "<|im_end|>\n<|im_start|>assistant\n"
    ),
}

MAX_TOKENS = 64


def cache_offsets(cache_list) -> list:
    """Return [layer.offset for each cache layer] — used for invariant checks.

    DFlashAttention / standard Attention caches expose `.offset` in the upstream
    contract (see `dflash_runtime.py:105` and mlx_lm cache utilities).
    """
    out = []
    for c in cache_list:
        out.append(getattr(c, "offset", None))
    return out


def baseline_decode(runner: MlxRunner, tokens: list[int], max_new: int, eos_id: int):
    """Greedy temp=0 decode via `prefill` + `decode_step`. Stops on EOS or max_new."""
    seq_id = 7001
    info = runner.prefill(seq_id, tokens)
    last = int(info["next_token"])
    position = int(info["position"])
    out = [last]
    transcript = [
        {"step": 0, "kind": "prefill", "tok": last, "pos_after": position}
    ]
    if last == eos_id:
        runner.remove_seq(seq_id)
        return out, transcript

    for step in range(1, max_new):
        res = runner.decode_step(seq_id, last, position)
        last = int(res["next_token"])
        position = int(res["position"])
        out.append(last)
        transcript.append(
            {"step": step, "kind": "decode", "tok": last, "pos_after": position}
        )
        if last == eos_id:
            break
    runner.remove_seq(seq_id)
    return out, transcript


def dflash_decode(runner: MlxRunner, tokens: list[int], max_new: int, eos_id: int):
    """DFlash speculative decode. Stops on EOS or max_new.

    Asserts cache invariants per block step:
      - target_cache offsets all equal `new_position` after rollback
      - draft_cache offsets all equal `new_position` after trim (when present)
    """
    seq_id = 7002
    info = runner.dflash_prefill(seq_id, tokens)
    last = int(info["next_token"])
    position = int(info["position"])
    out = [last]
    transcript = [
        {"step": 0, "kind": "prefill", "tok": last, "pos_after": position}
    ]

    target_cache, _ = runner.states[seq_id]
    draft_cache = runner.dflash_states[seq_id]["draft_cache"]
    t_off = cache_offsets(target_cache)
    d_off = cache_offsets(draft_cache)
    transcript[-1]["target_offsets"] = t_off
    transcript[-1]["draft_offsets"] = d_off

    if last == eos_id:
        runner.remove_seq(seq_id)
        return out, transcript

    block_idx = 0
    invariant_failures = []
    prev_position = position
    while len(out) < max_new:
        res = runner.dflash_block_step(seq_id, last)
        block_idx += 1
        accepted = int(res["accepted"])
        new_toks = [int(t) for t in res["new_tokens"]]
        new_position = int(res["position"])

        # cache invariants AFTER block step (post-rollback).
        # - target_cache (trim-able layer): offset == new_position. GDN layers
        #   in mlx_lm hybrid models expose an offset that we don't assert on
        #   (rollback is via gated_delta_update replay, not offset arithmetic).
        # - draft_cache: offset == prev_position (= new_position - accepted - 1).
        #   The draft consumed target_hidden of length (accepted_prev+1) as
        #   ctx_keys; the upstream-formula trim reduces offset back to the
        #   pre-block commit point. Equivalent to upstream's
        #   `cache.offset == prompt.size + n - 1` because in our framing
        #   `prompt.size + n - 1` ≡ prev_position.
        target_cache, _ = runner.states[seq_id]
        draft_cache = runner.dflash_states[seq_id]["draft_cache"]
        t_off = cache_offsets(target_cache)
        d_off = cache_offsets(draft_cache)
        for layer_idx, off in enumerate(d_off):
            if off is not None and off != prev_position:
                invariant_failures.append(
                    {
                        "block": block_idx,
                        "kind": "draft_offset_mismatch",
                        "layer": layer_idx,
                        "got": off,
                        "want_eq_prev_position": prev_position,
                        "new_position": new_position,
                    }
                )
        # Target trim-able layers: assert offset == new_position. We can't
        # cheaply distinguish trim-able from GDN here, so only assert on layers
        # where can_trim_prompt_cache returns True for THAT layer. mlx_lm
        # exposes per-cache-slice trimming via slicing the list.
        from mlx_lm.models.cache import can_trim_prompt_cache
        for layer_idx, off in enumerate(t_off):
            if off is None:
                continue
            try:
                trim_able = can_trim_prompt_cache([target_cache[layer_idx]])
            except Exception:
                trim_able = False
            if trim_able and off != new_position:
                invariant_failures.append(
                    {
                        "block": block_idx,
                        "kind": "target_offset_mismatch",
                        "layer": layer_idx,
                        "got": off,
                        "want": new_position,
                    }
                )

        # Append new_toks; respect EOS-inside-block (truncate at EOS, stop).
        eos_in_block = None
        for i, t in enumerate(new_toks):
            out.append(t)
            if t == eos_id:
                eos_in_block = i
                break
            if len(out) >= max_new:
                break

        transcript.append(
            {
                "step": block_idx,
                "kind": "block",
                "accepted": accepted,
                "new_tokens": new_toks,
                "pos_after": new_position,
                "target_offsets": t_off,
                "draft_offsets": d_off,
                "eos_in_block_index": eos_in_block,
            }
        )

        if eos_in_block is not None:
            break
        last = new_toks[-1]
        prev_position = new_position
        position = new_position

    runner.remove_seq(seq_id)
    return out, transcript, invariant_failures


def fmt_tok(tok: int, decode_fn) -> str:
    try:
        return repr(decode_fn([tok]))
    except Exception:
        return f"<id={tok}>"


def run_check(label: str, runner: MlxRunner, prompt: str, eos_id: int) -> dict:
    print(f"\n=== {label}: {prompt!r} ===")
    tok = runner.tokenizer
    ids = tok.encode(prompt)

    t0 = time.time()
    base_ids, base_transcript = baseline_decode(runner, ids, MAX_TOKENS, eos_id)
    base_t = time.time() - t0

    t0 = time.time()
    dflash_ids, dflash_transcript, invariants = dflash_decode(
        runner, ids, MAX_TOKENS, eos_id
    )
    dflash_t = time.time() - t0

    n_match = sum(
        1 for a, b in zip(base_ids, dflash_ids) if a == b
    )
    bit_identical = base_ids == dflash_ids

    base_text = tok.decode(base_ids)
    dflash_text = tok.decode(dflash_ids)

    n_blocks = sum(1 for e in dflash_transcript if e["kind"] == "block")
    accepts = [e["accepted"] for e in dflash_transcript if e["kind"] == "block"]
    avg_accept = sum(accepts) / max(len(accepts), 1)

    base_eos_idx = next(
        (i for i, t in enumerate(base_ids) if t == eos_id), None
    )
    dflash_eos_idx = next(
        (i for i, t in enumerate(dflash_ids) if t == eos_id), None
    )

    print(f"  baseline: {len(base_ids)} tokens in {base_t:.1f}s, eos@{base_eos_idx}")
    print(f"  dflash:   {len(dflash_ids)} tokens in {dflash_t:.1f}s, "
          f"{n_blocks} blocks, avg_accept={avg_accept:.2f}, eos@{dflash_eos_idx}")
    print(f"  bit-identical: {bit_identical}  match prefix: {n_match}/{min(len(base_ids), len(dflash_ids))}")
    print(f"  baseline text:  {base_text[:80]!r}")
    print(f"  dflash text:    {dflash_text[:80]!r}")
    if invariants:
        print(f"  ⚠ cache invariant failures: {len(invariants)} — {invariants[:3]}")
    else:
        print(f"  cache invariants: PASS (draft_offset == prev_position; trim-able target_offset == new_position)")

    return {
        "label": label,
        "prompt": prompt,
        "n_in_tokens": len(ids),
        "baseline": {
            "ids": base_ids,
            "text": base_text,
            "wall_s": round(base_t, 3),
            "eos_index": base_eos_idx,
        },
        "dflash": {
            "ids": dflash_ids,
            "text": dflash_text,
            "wall_s": round(dflash_t, 3),
            "n_blocks": n_blocks,
            "avg_accept": round(avg_accept, 3),
            "eos_index": dflash_eos_idx,
        },
        "bit_identical": bit_identical,
        "match_prefix_len": n_match,
        "cache_invariant_failures": invariants,
    }


def main():
    os.environ["LUMEN_MLX_SPEC"] = "dflash"
    os.environ.setdefault("LUMEN_MLX_DFLASH_MODEL", DRAFT_ID)

    runner = MlxRunner()
    info = runner.load(MODEL_ID)
    print(f"loaded: dflash enabled={info['dflash']['enabled']}")
    print(f"  draft={info['dflash']['draft_model_id']}")
    print(f"  block_size={info['dflash']['block_size']}")

    eos_candidates = []
    tok = runner.tokenizer
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try:
            ids = tok.encode(tname, add_special_tokens=False)
            if len(ids) == 1:
                eos_candidates.append((tname, ids[0]))
        except Exception:
            pass
    print(f"  eos candidates: {eos_candidates}")
    eos_id = eos_candidates[0][1] if eos_candidates else 248046

    results = []
    for label, prompt in PROMPTS.items():
        results.append(run_check(label, runner, prompt, eos_id))

    print("\n=== final summary ===")
    all_pass = True
    for r in results:
        status = "PASS" if r["bit_identical"] and not r["cache_invariant_failures"] else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  [{status}] {r['label']:6s} "
            f"baseline={len(r['baseline']['ids']):3d}tok eos@{r['baseline']['eos_index']} "
            f"dflash={len(r['dflash']['ids']):3d}tok eos@{r['dflash']['eos_index']} "
            f"blocks={r['dflash']['n_blocks']:3d} accept={r['dflash']['avg_accept']:.2f}"
        )

    out_json = {
        "summary": {
            "all_pass": all_pass,
            "model": MODEL_ID,
            "draft": DRAFT_ID,
            "max_tokens": MAX_TOKENS,
        },
        "results": results,
    }
    out_path = "/tmp/dflash_d2_harness.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nfull results → {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
