#!/usr/bin/env bash
# 100-batch sports team/league matching benchmark.
#
# Hits lumen-server directly (port 8080) since Moltis isn't an OpenAI-compatible
# proxy — it's a separate agent gateway. For programmatic batch matching the
# correct architecture is to bypass Moltis and call lumen-server's
# OpenAI-compatible /v1/chat/completions endpoint directly.
#
# Usage:
#   ./bench_sports_matching.sh            # 100 sequential requests, default endpoint
#   N=20 ./bench_sports_matching.sh       # 20 requests
#   ENDPOINT=http://other:8080 ./bench_sports_matching.sh
#
# Outputs:
#   /tmp/bench-results.jsonl   — one JSON line per request (query, response, latency_ms, ok)
#   /tmp/bench-summary.txt     — aggregate (count, ok-rate, mean/p50/p95/p99 latency, tok/s)

set -uo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8080}"
MODEL="${MODEL:-/Users/dev/models/gemma-4-26b-a4b-mlx-3bit}"
N="${N:-100}"
MAX_TOKENS="${MAX_TOKENS:-80}"

OUT=/tmp/bench-results.jsonl
SUM=/tmp/bench-summary.txt
: > "$OUT"
: > "$SUM"

# Test queries — mix of KBO baseball, K-League soccer, NBA, EPL, etc.
QUERIES=(
  "한화 이글스"  "두산 베어스"  "LG 트윈스"  "KIA 타이거즈"  "삼성 라이온즈"
  "SSG 랜더스"  "롯데 자이언츠"  "NC 다이노스"  "KT 위즈"  "키움 히어로즈"
  "전북 현대"  "울산 HD"  "FC 서울"  "수원 삼성"  "포항 스틸러스"
  "강원 FC"  "인천 유나이티드"  "대구 FC"  "광주 FC"  "제주 유나이티드"
  "Lakers"  "Celtics"  "Warriors"  "Heat"  "Knicks"
  "Bulls"  "Nets"  "Sixers"  "Bucks"  "Mavericks"
  "Manchester United"  "Liverpool"  "Arsenal"  "Chelsea"  "Manchester City"
  "Tottenham"  "Real Madrid"  "Barcelona"  "Bayern Munich"  "PSG"
  "Yankees"  "Red Sox"  "Dodgers"  "Giants"  "Cubs"
  "Patriots"  "Cowboys"  "Packers"  "49ers"  "Chiefs"
  "한화 이글스"  "두산 베어스"  "Yankees"  "Lakers"  "Manchester United"
  "LG 트윈스"  "Real Madrid"  "Celtics"  "Liverpool"  "FC 서울"
  "KIA 타이거즈"  "Warriors"  "Arsenal"  "Bayern Munich"  "Dodgers"
  "삼성 라이온즈"  "Heat"  "Chelsea"  "PSG"  "Giants"
  "SSG 랜더스"  "Knicks"  "Manchester City"  "Barcelona"  "Cubs"
  "롯데 자이언츠"  "Bulls"  "Tottenham"  "수원 삼성"  "Patriots"
  "NC 다이노스"  "Nets"  "포항 스틸러스"  "Cowboys"  "Red Sox"
  "KT 위즈"  "Sixers"  "강원 FC"  "Packers"  "Mavericks"
  "키움 히어로즈"  "Bucks"  "인천 유나이티드"  "49ers"  "전북 현대"
  "대구 FC"  "Chiefs"  "광주 FC"  "제주 유나이티드"  "울산 HD"
)

SYSTEM_PROMPT='You match user queries to sports teams. Reply with strict JSON only (no markdown fences): {"league": "...", "team": "...", "sport": "...", "country": "..."}'

echo "Starting benchmark: N=$N, endpoint=$ENDPOINT, model=$MODEL"
echo "Output: $OUT"
echo ""

t_start=$(python3 -c 'import time; print(time.time())')

ok=0
fail=0
total_tokens=0

for i in $(seq 0 $((N-1))); do
  q="${QUERIES[$((i % ${#QUERIES[@]}))]}"

  body=$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [
        {"role": "system", "content": sys.argv[2]},
        {"role": "user", "content": sys.argv[3]}
    ],
    "max_tokens": int(sys.argv[4]),
    "temperature": 0
}))
' "$MODEL" "$SYSTEM_PROMPT" "$q" "$MAX_TOKENS")

  t0=$(python3 -c 'import time; print(time.time())')
  resp=$(curl -s --max-time 30 -X POST "$ENDPOINT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body")
  t1=$(python3 -c 'import time; print(time.time())')

  lat_ms=$(python3 -c "print(int(($t1 - $t0) * 1000))")

  # Parse response: extract content + completion_tokens, check for error.
  parsed=$(echo "$resp" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    if "error" in d:
        print(json.dumps({"ok": False, "error": str(d["error"])[:200], "content": "", "completion_tokens": 0}))
    else:
        ch = d.get("choices", [{}])[0]
        msg = ch.get("message", {}).get("content", "")
        ct = d.get("usage", {}).get("completion_tokens", 0)
        print(json.dumps({"ok": True, "content": msg[:300], "completion_tokens": ct}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)[:200], "content": "", "completion_tokens": 0}))
' 2>/dev/null)

  parsed_ok=$(echo "$parsed" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("ok", False))')
  ct=$(echo "$parsed" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("completion_tokens", 0))')

  if [ "$parsed_ok" = "True" ]; then
    ok=$((ok+1))
    total_tokens=$((total_tokens + ct))
  else
    fail=$((fail+1))
  fi

  # Single jsonl record per request.
  python3 -c '
import json, sys
rec = {
    "i": int(sys.argv[1]),
    "query": sys.argv[2],
    "latency_ms": int(sys.argv[3]),
}
parsed = json.loads(sys.argv[4])
rec.update(parsed)
print(json.dumps(rec, ensure_ascii=False))
' "$i" "$q" "$lat_ms" "$parsed" >> "$OUT"

  # Live progress every 10 requests
  if [ $(( (i+1) % 10 )) -eq 0 ]; then
    echo "  [$((i+1))/$N] ok=$ok fail=$fail last_lat=${lat_ms}ms last_q=\"$q\""
  fi
done

t_end=$(python3 -c 'import time; print(time.time())')
elapsed=$(python3 -c "print(round($t_end - $t_start, 2))")

# Aggregate stats
python3 <<PYEOF | tee "$SUM"
import json, statistics

lats = []
oks = 0
fails = 0
total_completion_tokens = 0
with open("$OUT") as f:
    for line in f:
        r = json.loads(line)
        lats.append(r["latency_ms"])
        if r.get("ok"): oks += 1
        else: fails += 1
        total_completion_tokens += r.get("completion_tokens", 0)

lats_sorted = sorted(lats)
n = len(lats)
def pct(p):
    k = int(round(p/100.0 * (n - 1)))
    return lats_sorted[k]

print("=" * 60)
print(f"  BENCHMARK RESULTS — N={n}")
print("=" * 60)
print(f"  Elapsed wall:        {$elapsed} s")
print(f"  OK / FAIL:           {oks} / {fails}")
print(f"  OK rate:             {100*oks/n:.1f}%")
print(f"  Total tokens out:    {total_completion_tokens}")
print(f"  Mean tok/s (gen):    {total_completion_tokens / $elapsed:.2f}")
print()
print(f"  Latency (ms):")
print(f"    min   = {min(lats)}")
print(f"    mean  = {statistics.mean(lats):.0f}")
print(f"    p50   = {pct(50)}")
print(f"    p95   = {pct(95)}")
print(f"    p99   = {pct(99)}")
print(f"    max   = {max(lats)}")
print("=" * 60)
PYEOF

echo ""
echo "Per-request JSON: $OUT"
echo "Summary:          $SUM"
echo ""
echo "Sample responses (first 3):"
head -3 "$OUT" | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f"  Q={r[\"query\"]!r:30s} ({r[\"latency_ms\"]}ms) → {r.get(\"content\", \"\")[:120]!r}")
'
