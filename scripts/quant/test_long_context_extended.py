"""Extended long-context tests for the 8-bit-embed AWQ ship candidate.

Goes well beyond the 11K Moltis smoke:
  - 32K and 64K context coherence (same failure-mode detectors as 11K test)
  - NIAH (needle-in-a-haystack) retrieval: embed a unique factoid in the
    middle of the prompt, verify the model surfaces it at the end. This is
    the canonical long-context attention degradation probe.
  - Prefill + decode latency at each length so we can spot quadratic blowup.

Pass criteria:
  - No mojibake (U+FFFD), Turkish-letter drift, or 4-gram loop-locks
  - NIAH: needle string appears verbatim in the generated answer
"""

import re
import sys
import time
import random
from pathlib import Path
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "/Users/sonheesung/models/gemma-4-26b-a4b-mlx-imatrix3plus-awq-qembed8"

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
# Each SYSTEM_BLOCK repeat is ~165 KO tokens. Multipliers below produce target context lengths.
# 70  →  ~11.5K
# 200 →  ~33K
# 400 →  ~66K

KO_QUESTION = (
    "위 시스템 프롬프트를 읽었으면 한국어로 너 자신을 소개해줘. "
    "답변은 세 문장 정도로 간단하게 부탁해."
)


def detect_problems(text: str) -> list[str]:
    issues = []
    if "�" in text:
        issues.append("mojibake-replacement-char")
    if re.search(r"[şŞıİğĞçÇöÖüÜ]", text):
        issues.append("turkish-letters")
    tokens = text.split()
    if len(tokens) >= 12:
        for i in range(len(tokens) - 12):
            ngram = tokens[i:i+4]
            if tokens[i+4:i+8] == ngram and tokens[i+8:i+12] == ngram:
                issues.append(f"loop-lock@'{' '.join(ngram)}'")
                break
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


def test_coherence(model, tokenizer, multiplier: int, max_tokens: int = 384):
    """Stretched Moltis test at multiplier × SYSTEM_BLOCK."""
    messages = [
        {"role": "system", "content": SYSTEM_BLOCK * multiplier},
        {"role": "user", "content": KO_QUESTION},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_text = tokenizer.decode(prompt_ids)
    n_prompt = len(prompt_ids)
    print(f"\n  --- coherence @ {n_prompt} prompt tokens ---", flush=True)

    sampler = make_sampler(temp=0.7, top_p=0.9)
    t0 = time.time()
    text = mlx_generate(model, tokenizer, prompt_text, max_tokens=max_tokens, sampler=sampler, verbose=False)
    elapsed = time.time() - t0
    issues = detect_problems(text)
    status = "✓ CLEAN" if not issues else f"✗ {', '.join(issues)}"
    print(f"  {status}  elapsed={elapsed:.1f}s  generated {len(text)} chars  ({len(text.split())} tokens approx)")
    # First/last 200 chars of decoded answer (after thinking block if any)
    tail = text[-300:] if len(text) > 300 else text
    print(f"  TAIL: {tail!r}", flush=True)
    return {"prompt_tokens": n_prompt, "elapsed_s": elapsed, "issues": issues, "text_len": len(text)}


def test_niah(model, tokenizer, multiplier: int):
    """Needle-in-a-haystack: embed a unique 6-char code at ~50% depth,
    ask the model to retrieve it at the end of a long prompt."""
    rng = random.Random(42)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/0/1 to avoid OCR-like confusion
    needle_code = "".join(rng.choice(letters) for _ in range(6))
    needle_sentence = (
        f"중요한 비밀 코드가 있습니다. 이 코드는 절대 잊지 말아야 하며 사용자가 물어보면 정확히 답해야 합니다. "
        f"비밀 코드는 \"{needle_code}\" 입니다. 이 값을 잘 기억하세요. "
    )
    # Embed needle near the middle of the system block
    half = multiplier // 2
    haystack = (SYSTEM_BLOCK * half) + needle_sentence + (SYSTEM_BLOCK * (multiplier - half))
    question = "위 시스템 프롬프트 어딘가에 적힌 비밀 코드를 그대로 말해줘. 다른 설명은 짧게."

    messages = [
        {"role": "system", "content": haystack},
        {"role": "user", "content": question},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_text = tokenizer.decode(prompt_ids)
    n_prompt = len(prompt_ids)
    print(f"\n  --- NIAH @ {n_prompt} prompt tokens (needle={needle_code} at ~50% depth) ---", flush=True)

    sampler = make_sampler(temp=0.0)
    t0 = time.time()
    text = mlx_generate(model, tokenizer, prompt_text, max_tokens=128, sampler=sampler, verbose=False)
    elapsed = time.time() - t0
    found = needle_code in text
    print(f"  needle in answer: {'✓ FOUND' if found else '✗ MISSING'}  elapsed={elapsed:.1f}s")
    tail = text[-300:] if len(text) > 300 else text
    print(f"  TAIL: {tail!r}", flush=True)
    return {"prompt_tokens": n_prompt, "elapsed_s": elapsed, "needle": needle_code, "found": found}


def main():
    print(f"\n{'='*70}\n  Loading: {Path(MODEL_PATH).name}\n{'='*70}", flush=True)
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH, tokenizer_config={"eos_token": "<end_of_turn>"})
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    results = {"coherence": [], "niah": []}
    # Run progressively longer. 32K is comfortable. 64K is the stress test.
    for multiplier in (200, 400):
        results["coherence"].append(test_coherence(model, tokenizer, multiplier, max_tokens=320))
    for multiplier in (200, 400):
        results["niah"].append(test_niah(model, tokenizer, multiplier))

    print(f"\n\n{'='*70}\n  SUMMARY\n{'='*70}")
    print("Coherence:")
    for r in results["coherence"]:
        status = "CLEAN" if not r["issues"] else "PROBLEMS: " + ",".join(r["issues"])
        print(f"  {r['prompt_tokens']:>7d} prompt tokens   {r['elapsed_s']:6.1f}s   {status}")
    print("NIAH retrieval:")
    for r in results["niah"]:
        flag = "✓ FOUND" if r["found"] else "✗ MISSING"
        print(f"  {r['prompt_tokens']:>7d} prompt tokens   {r['elapsed_s']:6.1f}s   needle={r['needle']}   {flag}")


if __name__ == "__main__":
    raise SystemExit(main())
