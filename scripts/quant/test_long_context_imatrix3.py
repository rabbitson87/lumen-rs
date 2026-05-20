"""Real-failure-mode test: long Korean context (Moltis-class ~11K tokens).

The smoke script's `ko_long_ctx` is only ~3K tokens.  The original v0.1.1 / v0.1.2
failure on `gemma-4-26b-a4b-mlx-3bit` only triggered at production context length
(Moltis system prompt + multi-turn history ≈ 11K).  This script reproduces that
shape and tests both broken-3bit (control) and imatrix3 (treatment).

Pass criteria for imatrix3:
  - No catastrophic loops (no "비비-비비" or "karın-karın")
  - No Turkish-letter drift
  - No mojibake (U+FFFD replacement chars)
  - Coherent text in either Korean or English thinking + Korean response

We compare against the 4-bit ref to verify imatrix3 is on par.
"""

import sys
import re
import time
from pathlib import Path
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# Build a Moltis-like 11K-token system prompt
SYSTEM_BLOCK = (
    "당신은 Lumen이라는 로컬 LLM 추론 엔진의 어시스턴트입니다. "
    "사용자가 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답하세요. "
    "Lumen은 Apple Silicon에서 MLX 백엔드로 Gemma 4 / Qwen 3.6 모델을 서빙합니다. "
    "도구 호출은 JSON 형식이고 응답은 OpenAI 호환 형식입니다. "
    "당신은 친절하고 정확하며 사용자의 의도를 이해하려고 노력합니다. "
    "기술적 질문에는 정확한 답변을, 일상적 질문에는 자연스러운 한국어 답변을 제공하세요. "
    "Lumen 프로젝트는 Rust로 작성되었으며 atomic_http와 mlx-rs를 사용합니다. "
    "KV 캐시 최적화는 TurboQuant 기법을 통해 이루어지며 메모리 사용량을 크게 줄입니다. "
)
# Repeat to ~11K tokens.  Each repetition is ~1.0K tokens → 11 repetitions.

LONG_USER = (
    "위 시스템 프롬프트를 읽었으면 한국어로 너 자신을 소개해줘. "
    "답변은 세 문장 정도로 간단하게 부탁해."
)


def detect_problems(text: str) -> list[str]:
    """Catch the failure modes we care about."""
    issues = []
    # Mojibake / replacement char
    if "�" in text:
        issues.append("mojibake-replacement-char")
    # Turkish-specific letters (the original karın bug)
    if re.search(r"[şŞıİğĞçÇöÖüÜ]", text):
        issues.append("turkish-letters")
    # Repetition lock: same 4-gram repeated 3+ times in a row
    tokens = text.split()
    if len(tokens) >= 12:
        for i in range(len(tokens) - 12):
            ngram = tokens[i:i+4]
            if tokens[i+4:i+8] == ngram and tokens[i+8:i+12] == ngram:
                issues.append(f"loop-lock@'{' '.join(ngram)}'")
                break
    # Catastrophic character-level repetition (e.g. "비비-비비-비비")
    for length in (4, 6, 8):
        for i in range(len(text) - 3*length):
            chunk = text[i:i+length]
            if not chunk.strip():
                continue
            if text[i+length:i+2*length] == chunk and text[i+2*length:i+3*length] == chunk:
                issues.append(f"char-loop@'{chunk}'(len={length})")
                break
        if any("char-loop" in i for i in issues):
            break
    return issues


def run_model(model_path: str, max_tokens: int = 768, temperature: float = 0.7, top_p: float = 0.9) -> dict:
    print(f"\n{'='*70}\n  Loading: {Path(model_path).name}\n{'='*70}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path, tokenizer_config={"eos_token": "<end_of_turn>"})
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # SYSTEM_BLOCK is ~165 tokens after Korean BPE.  ~70 repetitions ≈ 11K.
    messages = [
        {"role": "system", "content": SYSTEM_BLOCK * 70},
        {"role": "user", "content": LONG_USER},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_text = tokenizer.decode(prompt_ids)
    print(f"  prompt tokens: {len(prompt_ids)}", flush=True)

    results = {}
    from mlx_lm.sample_utils import make_sampler
    for mode, sampler_kwargs in [
        ("greedy", {"temp": 0.0}),
        ("sampled", {"temp": temperature, "top_p": top_p}),
    ]:
        t0 = time.time()
        sampler = make_sampler(**sampler_kwargs)
        text = mlx_generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        issues = detect_problems(text)
        results[mode] = {
            "text": text,
            "issues": issues,
            "elapsed_s": time.time() - t0,
        }
        status = "✓ CLEAN" if not issues else f"✗ {', '.join(issues)}"
        print(f"  {mode:7s}  {status}   ({results[mode]['elapsed_s']:.1f}s)  generated {len(text)} chars")
        # Show start + middle + end to spot late-degeneration
        if len(text) > 900:
            preview = f"START: {text[:280]!r}\n              MID:   {text[len(text)//2-140:len(text)//2+140]!r}\n              END:   {text[-300:]!r}"
        else:
            preview = repr(text)
        print(f"  preview: {preview}", flush=True)

    return {"model": Path(model_path).name, "prompt_tokens": len(prompt_ids), **results}


def main():
    models = [
        "/Users/sonheesung/models/gemma-4-26b-a4b-mlx-4bit",          # reference (15 GB)
        "/Users/sonheesung/models/gemma-4-26b-a4b-mlx-imatrix3plus",  # top4=0.35 (12.5 GB)
        "/Users/sonheesung/models/gemma-4-26b-a4b-mlx-imatrix3",      # top4=0.25 (12 GB)
        "/Users/sonheesung/models/gemma-4-26b-a4b-mlx-3bit",          # control (11 GB, broken)
    ]
    summaries = []
    for path in models:
        try:
            res = run_model(path)
            summaries.append(res)
        except Exception as e:
            print(f"  ERROR on {path}: {e}", flush=True)
            summaries.append({"model": Path(path).name, "error": str(e)})

    print(f"\n\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"{'model':50s} {'greedy':30s} {'sampled':30s}")
    for s in summaries:
        if "error" in s:
            print(f"  {s['model']:50s} ERROR: {s['error']}")
            continue
        g_status = "CLEAN" if not s["greedy"]["issues"] else "PROB: " + ",".join(s["greedy"]["issues"])
        sa_status = "CLEAN" if not s["sampled"]["issues"] else "PROB: " + ",".join(s["sampled"]["issues"])
        print(f"  {s['model']:50s} {g_status:30s} {sa_status:30s}")


if __name__ == "__main__":
    raise SystemExit(main())
