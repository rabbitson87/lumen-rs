"""
Side-by-side smoke comparison of multiple MLX-quant variants of Gemma 4
26B-A4B against the bf16 reference. Designed to catch the long-context
language-drift bug class without needing a full perplexity sweep.

For each (quant) variant:
  1. Run the same N prompts at temperature=0 (greedy) for K tokens
  2. Run them again at temperature=0.7 (sampled) — same seed
  3. Record: tokens generated, EOS-terminated?, language stability,
     repetition lock-in detection (longest repeated 3-gram tail)

For each prompt, compare against the bf16 reference outputs token-by-token
to compute first-divergence-position. Outputs are written to a JSON
report so we can re-run after recipe tweaks and diff.

Smoke prompts are intentionally a mix of:
  - English short (low context)
  - Korean short (high-quant-sensitivity canary)
  - Long-context (>2K tokens) to surface accumulated quantization error

Usage:
  python scripts/quant/smoke_compare_quants.py \
      --ref /Users/sonheesung/models/gemma-4-26b-a4b-mlx-4bit \
      --candidates \
        /Users/sonheesung/models/gemma-4-26b-a4b-mlx-3bit-safe \
        /Users/sonheesung/models/gemma-4-26b-a4b-mlx-3bit \
      --out /tmp/lumen_quant_smoke.json
"""

import argparse
import json
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ─── Smoke prompts ─────────────────────────────────────────────────────
SMOKE_PROMPTS = [
    {
        "id": "ko_who",
        "lang": "ko",
        "context_kind": "short",
        "messages": [{"role": "user", "content": "넌 누구야?"}],
    },
    {
        "id": "ko_explain",
        "lang": "ko",
        "context_kind": "short",
        "messages": [
            {"role": "user", "content": "Apple Silicon에서 LLM 추론할 때 메모리 절약 팁 3개만 알려줘."}
        ],
    },
    {
        "id": "en_who",
        "lang": "en",
        "context_kind": "short",
        "messages": [{"role": "user", "content": "Who are you?"}],
    },
    {
        "id": "en_math",
        "lang": "en",
        "context_kind": "short",
        "messages": [
            {"role": "user", "content": "What is 1349 × 28? Show the steps."}
        ],
    },
    {
        "id": "ko_long_ctx",
        "lang": "ko",
        "context_kind": "long",
        # Long context = ~3K tokens of mixed Korean + English text, then a
        # short Korean question at the end. Mirrors Moltis's system-prompt
        # injection pattern (long English context + short Korean ask).
        "messages": [
            {
                "role": "system",
                "content": (
                    # Synthetic long context — repeated paragraph of Korean
                    # to stress quant noise accumulation. ~3K tokens.
                    "당신은 Lumen이라는 로컬 LLM 추론 엔진의 어시스턴트입니다. "
                    "사용자가 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답하세요. "
                    "Lumen은 Apple Silicon에서 MLX 백엔드로 Gemma 4 / Qwen 3.6 모델을 "
                    "서빙합니다. 도구 호출은 JSON 형식이고 응답은 OpenAI 호환 형식입니다. "
                ) * 30,
            },
            {"role": "user", "content": "넌 누구야?"},
        ],
    },
]


@dataclass
class PromptResult:
    id: str
    lang: str
    context_kind: str
    greedy_text: str = ""
    greedy_tokens: int = 0
    greedy_finish_reason: str = ""  # "eos" | "max_tokens" | "error"
    greedy_lang_drift: bool = False
    greedy_loop_lock: Optional[str] = None  # detected repeating n-gram tail
    sampled_text: str = ""
    sampled_tokens: int = 0
    sampled_finish_reason: str = ""
    sampled_lang_drift: bool = False
    sampled_loop_lock: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass
class QuantResult:
    model_path: str
    prompt_results: List[PromptResult] = field(default_factory=list)
    error: Optional[str] = None


def detect_loop_lock(text: str, n: int = 6, threshold: int = 4) -> Optional[str]:
    """Return the locked-in repeated phrase if `text` ends in N+ repetitions
    of the same short n-gram (whitespace-tokenized). Catches the "karın-"
    bug class. None if no lock-in.
    """
    if len(text) < 40:
        return None
    tail = text[-200:]
    # Match the trailing "X- X- X- ..." pattern by collapsing repeated
    # short tokens delimited by hyphen / space / punct.
    m = re.search(r"((\S{2,15}[-\s])\2{" + str(threshold - 1) + r",}\S*)\s*$", tail)
    if m:
        return m.group(2).strip("- ")
    # Also detect "word word word word ..." (no hyphen) repetitions.
    words = tail.split()
    if len(words) >= threshold * 2:
        last = words[-1]
        if all(w == last for w in words[-threshold:]):
            return last
    return None


def detect_lang_drift(prompt_lang: str, output: str) -> bool:
    """Crude check: if prompt is Korean and >30% of output bytes are clearly
    non-CJK ASCII letters in long runs (≥10 consecutive), flag drift. Same
    for English → CJK. Not perfect but catches the v0.1.1 / v0.1.2 case.
    """
    if not output.strip():
        return False
    if prompt_lang == "ko":
        ascii_run = re.search(r"[a-zA-Z]{15,}", output)
        # Bona-fide tech terms like "Apple Silicon" are OK — only flag
        # when a clear English sentence emerges.
        if ascii_run:
            # Count Hangul codepoints
            hangul = sum(1 for c in output if "가" <= c <= "힯")
            ascii_letters = sum(1 for c in output if c.isascii() and c.isalpha())
            if ascii_letters > hangul and ascii_letters > 40:
                return True
        # Turkish-letter detection (catches the original karın bug)
        if re.search(r"[şŞıİğĞçÇöÖüÜ]", output):
            return True
        return False
    if prompt_lang == "en":
        hangul = sum(1 for c in output if "가" <= c <= "힯")
        if hangul > 30:
            return True
        return False
    return False


def render_prompt(tokenizer, messages: List[dict]) -> str:
    """Apply chat template. Falls back to a manual format if the template
    fails (some MLX builds don't ship the jinja template intact).
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Manual fallback — Gemma 4 turn structure.
        out = ""
        for m in messages:
            role = m["role"]
            if role == "system":
                out += f"<start_of_turn>user\n{m['content']}<end_of_turn>\n"
            elif role == "user":
                out += f"<start_of_turn>user\n{m['content']}<end_of_turn>\n"
            elif role == "assistant":
                out += f"<start_of_turn>model\n{m['content']}<end_of_turn>\n"
        out += "<start_of_turn>model\n"
        return out


def run_prompt(model, tokenizer, prompt_def: dict, max_tokens: int) -> PromptResult:
    pr = PromptResult(
        id=prompt_def["id"],
        lang=prompt_def["lang"],
        context_kind=prompt_def["context_kind"],
    )

    prompt_text = render_prompt(tokenizer, prompt_def["messages"])

    # ── greedy ──
    t0 = time.time()
    try:
        greedy = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        pr.greedy_text = greedy
        # Token count is the difference in tokenizer length.
        full_ids = tokenizer.encode(prompt_text + greedy)
        prompt_ids = tokenizer.encode(prompt_text)
        pr.greedy_tokens = len(full_ids) - len(prompt_ids)
        pr.greedy_finish_reason = (
            "max_tokens" if pr.greedy_tokens >= max_tokens else "eos"
        )
        pr.greedy_lang_drift = detect_lang_drift(prompt_def["lang"], greedy)
        pr.greedy_loop_lock = detect_loop_lock(greedy)
    except Exception as e:
        pr.greedy_finish_reason = f"error: {e}"

    # ── sampled (T=0.7) ──
    try:
        sampled = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.7, top_p=0.9),
            verbose=False,
        )
        pr.sampled_text = sampled
        full_ids = tokenizer.encode(prompt_text + sampled)
        prompt_ids = tokenizer.encode(prompt_text)
        pr.sampled_tokens = len(full_ids) - len(prompt_ids)
        pr.sampled_finish_reason = (
            "max_tokens" if pr.sampled_tokens >= max_tokens else "eos"
        )
        pr.sampled_lang_drift = detect_lang_drift(prompt_def["lang"], sampled)
        pr.sampled_loop_lock = detect_loop_lock(sampled)
    except Exception as e:
        pr.sampled_finish_reason = f"error: {e}"

    pr.elapsed_ms = (time.time() - t0) * 1000
    return pr


def run_one_quant(model_path: str, max_tokens: int) -> QuantResult:
    result = QuantResult(model_path=model_path)
    print(f"\n══════════════════════════════════════════════════════")
    print(f"  Loading: {model_path}")
    print(f"══════════════════════════════════════════════════════")
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        result.error = f"load failed: {e}"
        print(f"  ✗ load error: {e}")
        return result

    for pdef in SMOKE_PROMPTS:
        print(f"  ▸ {pdef['id']} ({pdef['lang']}, {pdef['context_kind']}) ...")
        pr = run_prompt(model, tokenizer, pdef, max_tokens)
        result.prompt_results.append(pr)
        # Inline summary line per prompt.
        flags = []
        if pr.greedy_lang_drift:
            flags.append("G-DRIFT")
        if pr.greedy_loop_lock:
            flags.append(f"G-LOOP({pr.greedy_loop_lock[:20]})")
        if pr.sampled_lang_drift:
            flags.append("S-DRIFT")
        if pr.sampled_loop_lock:
            flags.append(f"S-LOOP({pr.sampled_loop_lock[:20]})")
        flag_str = ("  ⚠ " + " ".join(flags)) if flags else "  ✓ ok"
        print(
            f"    greedy:{pr.greedy_tokens}t/{pr.greedy_finish_reason}  "
            f"sampled:{pr.sampled_tokens}t/{pr.sampled_finish_reason}{flag_str}"
        )

    # release the model — 26B in memory is heavy
    del model
    del tokenizer
    mx.metal.clear_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        required=True,
        help="reference quant (typically the working 4-bit build) — same prompts "
        "compared against",
    )
    ap.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="quant paths to compare against the reference",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--out", default="/tmp/lumen_quant_smoke.json", help="JSON report path"
    )
    args = ap.parse_args()

    out: dict = {
        "max_tokens": args.max_tokens,
        "reference": args.ref,
        "candidates": [],
    }

    # Reference first
    ref_result = run_one_quant(args.ref, args.max_tokens)
    out["reference_result"] = {
        "model_path": ref_result.model_path,
        "error": ref_result.error,
        "prompt_results": [asdict(p) for p in ref_result.prompt_results],
    }

    for cand in args.candidates:
        cand_result = run_one_quant(cand, args.max_tokens)
        out["candidates"].append(
            {
                "model_path": cand_result.model_path,
                "error": cand_result.error,
                "prompt_results": [asdict(p) for p in cand_result.prompt_results],
            }
        )

    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[smoke] report written to {args.out}")

    # Verdict summary table
    print("\n┌──────────────────────────────┬────────┬────────────────────────────┐")
    print("│ prompt                       │ ref    │ candidates                 │")
    print("├──────────────────────────────┼────────┼────────────────────────────┤")
    for i, pdef in enumerate(SMOKE_PROMPTS):
        ref_p = ref_result.prompt_results[i] if i < len(ref_result.prompt_results) else None
        line = f"│ {pdef['id']:28s} │ "
        if ref_p:
            ref_status = "DRIFT" if ref_p.greedy_lang_drift or ref_p.sampled_lang_drift else "ok"
            line += f"{ref_status:6s} │ "
        else:
            line += f"-      │ "
        for cand in out["candidates"]:
            if i < len(cand["prompt_results"]):
                p = cand["prompt_results"][i]
                status = "DRIFT" if p["greedy_lang_drift"] or p["sampled_lang_drift"] else "ok"
                if p["greedy_loop_lock"] or p["sampled_loop_lock"]:
                    status = "LOOP"
                line += f"{status:6s} "
        line += "│"
        print(line)
    print("└──────────────────────────────┴────────┴────────────────────────────┘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
