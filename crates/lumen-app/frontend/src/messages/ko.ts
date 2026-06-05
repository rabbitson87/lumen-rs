/**
 * 한국어 메시지 사전. 키는 en.ts와 동일하게 유지한다 — 누락 키는
 * 자동으로 영어 fallback으로 채워지므로 부분 번역도 안전하다.
 */
export const ko: Record<string, string> = {
  // ── 탭바 ────────────────────────────────────────────────────────
  "tabs.main": "모델 & 서버",
  "tabs.tuning": "튜닝",
  "tabs.api": "API",
  "tabs.debug": "디버그",
  "tabs.language": "언어",

  // ── 헤더 ────────────────────────────────────────────────────────
  "header.start": "시작",
  "header.stop": "중지",
  "header.starting": "시작 중…",
  "header.stopping": "중지 중…",
  "header.logs": "로그",
  "header.env": "환경 변수",
  "header.doctor": "진단",
  "header.update": "업데이트",
  "header.title.brokenActive":
    "활성 모델의 다운로드가 완료되지 않았습니다. 모델 카드에서 재다운로드를 먼저 진행하세요.",
  "header.title.outdatedActive":
    "활성 모델의 최신 버전이 Hub에 있습니다. 모델 카드에서 업데이트를 먼저 진행하세요.",

  // ── 상태 표시 ───────────────────────────────────────────────────
  "status.stopped": "중지됨",
  "status.starting": "시작 중",
  "status.running": "실행 중",
  "status.stopping": "중지 중",
  "status.crashed": "비정상 종료",

  // ── 공용 액션 ───────────────────────────────────────────────────
  "action.download": "다운로드",
  "action.delete": "삭제",
  "action.use": "사용",
  "action.update": "업데이트",
  "action.redownload": "재다운로드",
  "action.reset": "초기화",
  "action.cancel": "취소",
  "action.confirm": "확인",
  "action.close": "닫기",
  "action.openConfig": "설정 폴더 열기",

  // ── 모델 카드 ───────────────────────────────────────────────────
  "models.title": "모델",
  "models.empty.unsupported":
    "지원되는 로컬 모델이 없습니다. 아래 추천 목록에서 받아주세요.",
  "models.empty.none": "로컬 모델이 없습니다. 아래 추천 목록에서 받아주세요.",
  "models.thisMac": "이 Mac:",
  "models.thisMac.ramSuffix": "GB RAM",
  "models.thisMac.overflow": "— 이 용량을 초과하는 모델은 표시됩니다.",
  "models.picker.placeholder": "— 추천 모델 선택 —",
  "models.picker.allDownloaded": "추천 모델 모두 다운로드 완료",
  "models.broken.label": "⚠ 다운로드 미완료 — 사용 전 재다운로드 필요",
  "models.outdated.label": "⚠ Hub에 최신 가중치 있음 — 사용 전 업데이트 필요",
  "models.unsupported.label": "지원 카탈로그 외 모델",

  // ── 서버 카드 ───────────────────────────────────────────────────
  "server.title": "서버",
  "server.cors": "CORS",
  "server.cors.off": "off (특정 IP)",
  "server.cors.localhost": "localhost (127.0.0.1)",
  "server.cors.all": "all / 0.0.0.0 (위험)",
  "server.host": "호스트",
  "server.port": "포트",
  "server.apiKey": "API 키",
  "server.apiKey.hint": "→ API 카드에서 설정",
  "server.memory.title": "Metal 메모리",
  "server.memory.titleHint": "(mlx-native)",
  "server.memory.tunedFor": "최적화 대상:",
  "server.memory.systemDefault": "기본 (시스템):",
  "server.memory.wired": "Wired GB",
  "server.memory.cache": "Cache GB",
  "server.memory.memory": "Memory GB",
  "server.memory.explainer": "각 항목 설명",

  // ── 지표 카드 ───────────────────────────────────────────────────
  "metrics.title": "지표",
  "metrics.tokensPerSec": "tok/s",
  "metrics.msPerStep": "ms / step",
  "metrics.kvCache": "KV 캐시",
  "metrics.requestsPerMin": "req/min",

  // ── 컨텍스트 카드 ───────────────────────────────────────────────
  "context.title": "컨텍스트",
  "context.titleHint": "(QUANT 상태에 연동)",
  "context.max": "최대",
  "context.sliding": "슬라이딩",
  "context.prefill": "프리필",
  "context.defaultMaxTokens": "기본 max_tokens",
  "context.kvQuant.label": "캐시 모드:",
  "context.kvQuant.offHint": "· 기본 KV 메모리 (압축 없음)",
  "context.recommended": "이 Mac 권장 최대",
  "context.recommended.suffix": "토큰",
  "context.warn.turnOnKvQuant":
    "— 더 긴 컨텍스트는 캐시 양자화를 켜야 안전합니다",

  // ── 양자화 (튜닝 탭) ────────────────────────────────────────────
  "quant.title": "캐시",
  "quant.titleHint": "(KV 캐시 양자화)",
  "quant.mode": "캐시 모드",
  "quant.mode.off": "끔",
  "quant.mode.on": "켬",
  "quant.mode.auto": "자동",
  "quant.autoThreshold": "자동 임계값 (토큰)",
  "quant.bits": "비트",
  "quant.on": "켜짐",
  "quant.off": "꺼짐",

  // ── 디버그 카드 ─────────────────────────────────────────────────
  "debug.title": "디버그",
  "debug.titleHint": "(긴급 점검용 스위치)",
  "debug.intro":
    "정상 동작이 실패할 때 사용하는 탈출구입니다. 평소엔 비워두거나 꺼두세요 — 실제로 영향을 주는 값(모델, 메모리 캡, 백엔드)은 모델 & 서버 탭에 있습니다.",
  "debug.memoryBypass": "메모리 우회",
  "debug.memoryBypass.label": "모든 캡 무시",
  "debug.memoryBypass.hint": "wired+cache+memory를 모두 우회, MLX/macOS에 위임",
  "debug.loader": "로더 오버라이드",
  "debug.tokenizer": "토크나이저",
  "debug.tokenizer.placeholder": "HF repo id (오버라이드)",
  "debug.weightsDir": "가중치 경로",
  "debug.weightsDir.placeholder": "활성 모델에서 자동 설정",
  "debug.skipWarmup": "워밍업 생략",
  "debug.skipWarmup.hint": "부팅 빠름, 첫 요청 느림",

  // ── 언어 탭 ─────────────────────────────────────────────────────
  "language.title": "언어",
  "language.choose": "인터페이스 언어",
  "language.note":
    "언어 설정은 브라우저에 로컬 저장되며 즉시 적용됩니다 — 재시작 필요 없습니다.",

  // ── 확인 모달 ───────────────────────────────────────────────────
  "confirm.delete.chat.title": "모델을 삭제할까요?",
  "confirm.delete.embedding.title": "임베딩 모델을 삭제할까요?",
  "confirm.delete.warning":
    "가중치가 디스크에서 삭제됩니다. 나중에 카탈로그에서 다시 다운로드할 수 있습니다.",
  "confirm.delete.embedding.activeNote": " 활성 임베딩이 해제됩니다.",
  "confirm.delete.busy": "삭제 중…",

  // ── 푸터 탭 (하단 도크) ─────────────────────────────────────────
  "footer.logs": "로그",
  "footer.env": "환경 변수 오버라이드",
  "footer.doctor": "진단",
  "footer.update": "업데이트",
  "footer.logs.empty":
    "아직 로그 출력이 없습니다. 서버를 시작하면 디코드/인코드 추적이 표시됩니다.",

  // ── 캐시(KV 양자화) 카드 툴팁 ───────────────────────────────────
  "quant.tooltip.mode":
    "끔: KV를 절대 압축하지 않음 (모든 디코드가 가장 빠름, 메모리는 가장 큼). 켬: 항상 압축 (KV 메모리 4–5× 절감, 디코드 ≈ 같음 ±5%). 자동: 이번 요청의 prompt가 아래 임계값 이상일 때만 압축 — 짧은 대화는 풀 속도, 긴 컨텍스트만 메모리 절감. 요청별 결정은 `[gemma4-backend] quant_kv_auto: ...` 로 로그에 기록됩니다.",
  "quant.tooltip.autoThreshold":
    "자동 모드가 KV 양자화를 켜는 prompt 토큰 수. 기본 131072 (128K) — Apple Silicon 통합 메모리가 bf16 KV를 충분히 담을 수 있는 범위 (M3 Max 36 GB에서 64K bf16 KV ≈ 16 GB) 아래에서는 압축 없이 풀 속도, 이를 넘으면 메모리 천장에 닿기 전에 압축으로 전환합니다.",
  "quant.tooltip.bits":
    "KV 채널당 양자화 비트 수. 8: 최고 품질, bf16 대비 2× 축소. 6: 균형 ≈ 2.7× 축소. 4: 권장 기본값, 4× 축소. 3: 최대 압축 약 5.3×, 미세한 품질 저하. mlx affine 양자화 (group_size=64) 를 사용하므로 별도의 회전/잔차 보정이 없습니다.",

  // ── CONTEXT 카드 설명 ───────────────────────────────────────────
  "context.hint.max.prefix":
    "최대 시퀀스 길이 (토큰). 호스트 RAM이 모델의 네이티브 한계를 감당할 수 없을 때 모델의 max_position_embeddings를 제한합니다 (Gemma 4는 128K를 표방).",
  "context.hint.max.kvOn":
    "현재 캐시 양자화 설정으로 대략 위에 표시된 KV 압축률 적용",
  "context.hint.max.kvOnRealistic": "— 이 Mac에서의 실질 한계:",
  "context.hint.max.kvOff":
    "캐시 양자화 꺼짐 — KV는 bf16 유지, 이 Mac에서의 실질 한계는",
  "context.hint.max.kvOffFallback": "모델 네이티브 최대치보다 훨씬 낮습니다",
  "context.hint.max.env": "환경 변수:",
  "context.hint.sliding":
    "슬라이딩 윈도우 어텐션 크기. 일부 레이어 (Gemma 4: 30개 중 25개)는 전체 시퀀스 대신 최근 N개 토큰에만 어텐션을 적용합니다 → 긴 컨텍스트에서 KV 메모리가 유한. 0 = 모델 내장 기본값 사용, N>0이면 오버라이드 (작을수록 KV 절감, 장거리 회상 약화).",
  "context.hint.sliding.kvStacks":
    "캐시 양자화와 함께 적용됩니다 — 슬라이딩은 어떤 토큰을 유지할지 결정, 양자화는 어떻게 저장할지 결정.",
  "context.hint.prefill":
    "프롬프트 처리 청크 상한. 이 값보다 긴 프롬프트는 \"prompt too large\" 오류로 거부됩니다. 클수록 긴 프롬프트를 받지만 프리필 동안 피크 메모리도 증가 (어텐션 QK·T = 청크 × KV",
  "context.hint.defaultMaxTokens":
    "OpenAI 호환 chat/completion 요청에서 `max_tokens`가 빠졌을 때 서버가 적용하는 생성 토큰 예산 + 클라이언트가 `max_tokens`를 명시한 경우의 상한선. 예: 8192로 설정하면 클라이언트가 더 큰 값(204800 등)을 보내도 8192로 cap됩니다. `0`은 상한 없음 (EOS/컨텍스트까지 무제한 — runaway CoT 주의). 환경 변수 `LUMEN_DEFAULT_MAX_TOKENS`와 `LUMEN_MAX_TOKENS_CAP` 양쪽으로 emit됩니다.",

  // ── SERVER 메모리 설명 ──────────────────────────────────────────
  "server.memory.explainer.intro":
    "Apple Silicon은 CPU와 GPU가 하나의 RAM 풀을 공유합니다. 아래 세 가지 상한은 MLX가 그 풀의 얼마만큼을 사용할 수 있는지 알려줍니다:",
  "server.memory.explainer.wired":
    "Wired GB — GPU에 고정되어 페이지 아웃되지 않는 RAM. LUMEN_WIRED_LIMIT_BYTES를 통해 활성 모델의 safetensors 바이트 크기에 정확히 맞춰 자동 설정되므로, 14.45 GB 모델이 14 GB 상한에서 잘리지 않습니다. KV 캐시 헤드룸이 더 필요하면 직접 입력해 오버라이드하세요.",
  "server.memory.explainer.cache":
    "Cache GB — MLX의 일시 버퍼 재사용 풀 (활성화값, 임시 영역). 작은 고정 예산 (2 GB)이면 충분합니다 — 시스템 RAM에 비례시키면 OS에 돌려줄 메모리를 미리 점유할 뿐입니다.",
  "server.memory.explainer.memory":
    "Memory GB — Metal 할당의 소프트 총량 상한. 이 값에 도달하면 단단한 wired 한계 전에 캐시 축출이 트리거됩니다. 모델 크기 + 2 GB + KV 캐시 예산 (≈ ctx ÷ 8K) 으로 설정하세요.",
  "server.memory.wired.titleHint":
    "LUMEN_WIRED_LIMIT_BYTES — 정확한 safetensors 크기",
  "server.host.title.pin": "특정 IP로 고정",
  "server.host.title.auto": "CORS 범위에 따라 자동 설정",

  // ── 모델 카드 추가 ──────────────────────────────────────────────
  "models.action.title.redownload":
    "누락 또는 잘린 파일을 검증 후 재다운로드",
  "models.action.title.update": "최신 Hub 가중치로 재다운로드",
  "models.action.title.unsupported": "서버 측 지원 카탈로그에 없음",

  // ── 헤더 메모리 바 ──────────────────────────────────────────────
  "header.memory.title": "시스템 메모리 — wired + active + compressor",
  "header.statusError": "· 오류",
  "header.doctor.title.idle": "프리플라이트 점검 실행",

  // ── API 탭 (ApiTabs.svelte) ─────────────────────────────────────
  "api.title": "API",
  "api.style.openai": "OpenAI 스타일",
  "api.style.claude": "Claude 스타일",
  "api.serverNotRunning":
    "서버가 {state} 상태입니다 — 시작 누르면 클라이언트가 사용할 값입니다",
  "api.baseUrl": "베이스 URL",
  "api.apiKey": "API 키",
  "api.apiKey.placeholder": "(없음 — 인증 비활성)",
  "api.copy": "복사",
  "api.copied": "복사됨",
  "api.copyFailed": "복사 실패:",
  "api.endpoints": "엔드포인트",
  "api.curlExample": "curl 예제",
  "api.anthropicVersion": "anthropic-version",
  "api.embedding.title": "임베딩 모델",
  "api.embedding.empty":
    "다운로드된 임베딩 모델이 없습니다 — 아래에서 먼저 받아주세요.",
  "api.embedding.disable": "임베딩 비활성화",
  "api.embedding.disable.title":
    "임베딩 모델 사용 중단 (/v1/embeddings 비활성)",
  "api.embedding.activeMissing.prefix": "활성 임베딩",
  "api.embedding.activeMissing.suffix": "이(가) 로컬 카탈로그에 없습니다.",
  "api.embedding.download.title": "임베딩 다운로드",
  "api.embedding.download.placeholder": "— 다운로드할 항목 선택 —",
  "api.embedding.endpointRequiresEmbedding": "(임베딩 모델 필요)",

  // ── 캐시 비트 비교 설명 ─────────────────────────────────────────
  "quant.hint.kvOff":
    "캐시 양자화 꺼짐 — KV는 bf16 유지 (최고 속도, 최대 메모리 사용)",
  "quant.hint.smallerVsFp16": "× bf16 대비 축소",
  "quant.hint.lowestMemory": "최저 메모리 사용, 미세한 품질 저하",
  "quant.hint.balancedQuality": "메모리/품질 균형",
  "quant.hint.highestQuality": "bf16에 가장 가까운 품질",
  "quant.hint.baseline": "권장 기본값 (4× 메모리 절감)",

  // ── CONTEXT 배너 ────────────────────────────────────────────────
  "context.banner.smallerThanBf16": "× bf16 대비 축소",
  "context.banner.kvCache": "KV 캐시 약 ",

  // ── 모델 카드 임베딩 미니행 ─────────────────────────────────
  "models.embedding.label": "임베딩:",
  "models.embedding.disabled": "(비활성)",
  "models.embedding.autoFetched": "(서버가 캐시에서 자동 로드)",
  "models.embedding.pickHint": "API 탭에서 선택 / 변경",

  // ── 환경 변수 패널 ──────────────────────────────────────────────
  "env.title": "환경 변수 오버라이드",
  "env.intro":
    "각 행은 서버가 시작 시 읽는 환경 변수 1개에 매핑됩니다. 문자열만 가능 — 불리언은 \"1\" 또는 \"0\"으로 입력.",
  "env.placeholder.key": "키",
  "env.placeholder.value": "값",
  "env.empty": "설정된 오버라이드가 없습니다. 아래에서 행을 추가하세요.",
  "env.add": "행 추가",
  "env.save": "저장",
  "env.savedHint": "저장됨. 적용하려면 서버 재시작 (중지 → 시작).",
  "env.knownKeysTitle": "알려진 키 (자동완성):",
  "env.intro2":
    "lumen-server 하위 프로세스에 전달되는 타입화된 노브. 토글하거나 값을 선택하거나, × 로 지우면 기본값으로 복구됩니다. UI 필드와 동일한 키는 강조 표시됩니다.",
  "env.row.remove": "제거",
  "env.row.shadowsPrefix": "⚠ 다음 UI 필드를 가립니다:",
  "env.add2": "+ 추가",
  "env.revert": "되돌리기",
  "env.applyHint": "변경 사항은 다음 서버 시작 시 적용됩니다.",
  "env.empty2": "설정된 오버라이드가 없습니다.",

  // ── 섹션 그루핑 ────────────────────────────────────────────
  "env.section.thinking": "추론",
  "env.section.sampling": "샘플링",
  "env.section.safety": "안전망",
  "env.section.advanced": "고급",
  "env.section.debug": "디버그 / 진단",

  // ── 항목별 라벨 (key = env 변수명) ──────────────────────
  "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.label": "백엔드 씽킹 기본값",
  "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.help":
    "ON 일 때 OpenAI 호환 클라이언트가 per-request 씽킹 신호 안 보내면 기본으로 씽킹 활성화. per-request 신호가 우선합니다.",

  "env.entry.LUMEN_TEMPERATURE.label": "온도 (Temperature)",
  "env.entry.LUMEN_TEMPERATURE.help":
    "Gemma 4 샘플링 온도. 클라이언트가 미지정(서버 기본 0.7)이면 이 값으로 치환, 명시하면 그 값 존중. Ollama gemma4 = 1.0. gemma4 한정.",

  "env.entry.LUMEN_TOP_P.label": "Top-p (nucleus)",
  "env.entry.LUMEN_TOP_P.help":
    "누적 확률 컷오프. 클라이언트 미지정 시 적용되는 기본값. Ollama gemma4 = 0.95.",

  "env.entry.LUMEN_TOP_K.label": "Top-k",
  "env.entry.LUMEN_TOP_K.help":
    "상위 k 개 토큰만 후보로 제한. 0 = 비활성화. Ollama gemma4 = 64. 너무 낮으면 반복 사이클 탈출 실패.",

  "env.entry.REPEAT_PENALTY.label": "반복 페널티",
  "env.entry.REPEAT_PENALTY.help":
    "최근 토큰 재등장에 페널티. 1.0 = off. Ollama = 1.1.",

  "env.entry.LUMEN_DRY_MULTIPLIER.label": "DRY 반복 억제",
  "env.entry.LUMEN_DRY_MULTIPLIER.help":
    "DRY(Don't Repeat Yourself) 강도. 0 = off. 0.8 권장 — }}}}/~~~~ 같은 섞인 degenerate 폭주를 직접 억제 (멀티턴 안정성).",

  "env.entry.LUMEN_MAX_THINKING_TOKENS.label": "최대 씽킹 토큰 (하드캡)",
  "env.entry.LUMEN_MAX_THINKING_TOKENS.help":
    "N 토큰 이상 추론 시 채널 종료 강제. 0 = 비활성화. Gemma 4 권장 600.",

  "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.label": "강제 종료 시도 횟수",
  "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.help":
    "턴 종료 전 채널 종료 토큰을 강제하는 횟수. 기본 1.",

  "env.entry.LUMEN_RUNAWAY_DETECT.label": "N-gram 폭주 감지기",
  "env.entry.LUMEN_RUNAWAY_DETECT.help":
    "n-gram 사이클 감지 시 응답 자동 절단. 기본 ON.",

  "env.entry.LUMEN_RUNAWAY_NGRAM.label": "폭주 n-gram 크기",
  "env.entry.LUMEN_RUNAWAY_NGRAM.help":
    "감지기가 반복 검사하는 n-gram 길이. 기본 4.",

  "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.label": "폭주 최대 반복",
  "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.help":
    "절단 전 허용되는 동일 n-gram 반복 횟수. 기본 8.",

  "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.label": "Phase B 로짓 보정 (Gemma 4)",
  "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.help":
    "사이드카 Δ 를 7 개 핵심 토큰 (채널/턴/툴 경계) 에 적용. 기본 ON.",

  "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.label": "Lark 문법 (Gemma 4 툴콜)",
  "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.help":
    "Lark 문법으로 툴콜 출력 강제 — call:NAME{...} 구조 무조건 보장. 기본 ON; free-form 툴콜 emission 필요 시에만 OFF.",

  "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.label": "tool_choice auto → required 승격",
  "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.help":
    "클라이언트가 tool_choice=\"auto\" 를 보낼 때 required 로 승격 → 매 턴 tool call 강제. Ayla 등 agentic 루프에서 모델이 task_complete 를 약하게 emit 해 응답이 여러 번 반복될 때 ON. 답변 텍스트는 tool 의 summary 필드에 담김. 기본 OFF.",
  "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.label": "thinking OFF 시 빈 thought 채널 주입",
  "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.help":
    "thinking OFF 일 때 생성 프롬프트에 빈 <|channel>thought<channel|> 블록을 주입할지. jinja 템플릿은 주입하지만 Ollama 의 native gemma4 렌더러는 주입 안 함(emptyBlockOnNothink=false). 주입하면 양자화 모델이 문장 도중 <turn|> 로 조기 종료하고 task_complete 를 안 부름. 기본 OFF(=Ollama 동작, 권장). 옛 jinja 동작 복원하려면 ON.",

  "env.entry.LUMEN_USE_JINJA_RENDERER.label": "minijinja 렌더러 사용 (Gemma 4)",
  "env.entry.LUMEN_USE_JINJA_RENDERER.help":
    "모델의 chat_template.jinja 를 minijinja 로 직접 렌더링 (Rust hand-port 대신). 골든 벡터에서 byte-identical 검증 완료 — 원본 jinja 가 정답인 path 로 전환하려면 ON.",

  "env.entry.LUMEN_DUMP_PROMPT.label": "프롬프트 덤프",
  "env.entry.LUMEN_DUMP_PROMPT.help":
    "모델에 보내는 chat template 적용 프롬프트 출력. off / preview / full.",

  "env.entry.LUMEN_LOG_REQUEST_BODY.label": "요청 본문 로깅",
  "env.entry.LUMEN_LOG_REQUEST_BODY.help":
    "/v1/chat/completions 요청마다 [diag] 한 줄 요약 출력.",

  "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.label": "토큰별 트레이스 (Gemma 4)",
  "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.help":
    "샘플링된 토큰마다 [token-trace] 한 줄 출력. 출력량 큼 — 디버그 전용.",

  "env.entry.LUMEN_EOS_GUARD_VERBOSE.label": "EOS 가드 상세 로그",
  "env.entry.LUMEN_EOS_GUARD_VERBOSE.help":
    "샘플링 파이프라인에서 발생하는 모든 EOS 가드 억제 이벤트 로깅.",

  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.label": "Qwen prefill 청크 크기",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.help":
    "Qwen 3.6 장문 프롬프트의 prefill 청크당 토큰 수. 클수록 GPU 동기화 횟수 ↓(cold prefill 빠름)이지만 peak 메모리 ↑. RAM 여유 있으면 키우고, 장문서 OOM 나면 낮추세요. 기본 2048.",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.label": "Qwen prefill 청크 로그",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.help":
    "청크별 prefill 시간과 Metal peak 메모리 출력. 디버그 전용.",
  "env.entry.LUMEN_NATIVE_TIMING.label": "네이티브 단계 타이밍",
  "env.entry.LUMEN_NATIVE_TIMING.help":
    "네이티브 MLX 러너의 단계별 forward 시간(embed / attention / linear-attn / MoE / lm_head ms) 출력. prefill/decode 병목 탐색용. ⚠️ 프로파일링 전용 — 레이어마다 GPU 동기화 배리어를 삽입해 MLX 파이프라이닝을 깨고 처리량을 ~8배 떨어뜨림(예: ~70→~8 tok/s). 평상시엔 반드시 OFF.",
  "env.entry.LUMEN_QWEN35_TOOL_DEBUG.label": "Qwen 툴콜 raw 덤프",
  "env.entry.LUMEN_QWEN35_TOOL_DEBUG.help":
    "Qwen3.6 툴콜 턴의 모델 raw 출력 전체(<tool_call>…</tool_call> 원문) 출력. 누락/유실된 툴 인자 진단용. 장황함 — 디버그 전용.",
  "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.label": "필수 툴 파라미터 강제",
  "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.help":
    "Qwen3.6 툴콜에서 required 파라미터가 빠진 채 함수를 닫기 전에 <parameter=KEY> opener를 주입 — path 없는 read() 같은 빈 호출 방지. 값은 모델이 직접 작성. 툴 많고 약한/양자화 모델에 유효; 기본 꺼짐.",
  "env.entry.LUMEN_QWEN35_TQ_KV.label": "TurboQuant KV 캐시",
  "env.entry.LUMEN_QWEN35_TQ_KV.help":
    "Qwen3.6 full-attention KV 캐시를 TurboQuant(회전 + Lloyd-Max 스칼라 양자화)으로 압축 — 커지는 KV 메모리를 ~2-4× 줄여 더 긴 컨텍스트를 OOM 전에 수용. dequant-on-read 비용이 약간 있지만, 장문에선 절감 메모리가 이를 상쇄할 수 있음. linear-attention 레이어는 영향 없음. 실험적 — 의존 전 품질(cosine/top-1) 측정 권장. 기본 꺼짐.",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.label": "TurboQuant KV 비트",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.help":
    "TurboQuant KV의 Lloyd-Max 비트 폭(2-8). 8 = 거의 무손실(여기서 시작), 6 = 최저 clean, 4 = 공격적/손실. TurboQuant KV 캐시 활성 시에만 사용. 비트 낮을수록 메모리 절감 크지만 품질 손실 증가.",

  // 메모리 계산기 (context / chunk / KV 모드별 peak 메모리 예측).
  "memcalc.title": "메모리 계산기",
  "memcalc.noGeometry":
    "모델 '{model}'의 메모리 프로파일이 없습니다. 카탈로그의 네이티브 MLX 모델(예: Qwen3.6-35B)을 지원합니다.",
  "memcalc.budget": "예산",
  "memcalc.budget.hint":
    "MLX 메모리 예산 ≈ 머신 RAM − OS 여유분. 정확한 값은 시작 로그 [mlx-mem] memory_limit set to N GB 에서 확인.",
  "memcalc.kv": "KV",
  "memcalc.bits": "bits",
  "memcalc.bits.hint":
    "bits = 품질만 (LUMEN_QWEN35_TQ_KV_BITS). 코드를 unpacked(uint8)로 저장해 어떤 bits든 메모리는 ~2×. 8 = 거의 무손실, 6 = 최저 clean, 4 = 손실. 4-bit에서 'uint4 packed' 체크 시 진짜 ~4× 미리보기 (패킹 미연결).",
  "memcalc.packed": "uint4 packed*",
  "memcalc.peak": "peak / 예산",
  "memcalc.chunk": "청크",
  "memcalc.prefix": "prefix 캐시",
  "memcalc.context": "컨텍스트",
  "memcalc.over": "예산 초과",
  "memcalc.maxAtConfig": "이 설정의 최대 컨텍스트:",
  "memcalc.table.title": "최대 컨텍스트 (토큰) — 예산 {budget} GB",
  "memcalc.table.note":
    "청크 축소가 가장 큰 레버(attention score가 청크×컨텍스트로 증가). TQ는 persistent KV를 줄임. *TQ4 = uint4 패킹, 미연결(미리보기). 추정치 — 본인 peak= 로그로 보정 권장.",

  // QUANT / CONTEXT / SERVER 카드 저장 시 공통 토스트. 서버 실행 중이면
  // 재시작 안내, 정지 상태면 단순 "저장됨" 만 표시. env-derived 노브
  // (캐시 모드/비트, ctx caps 등) 는 재시작 시점에만 반영됨.
  "config.savedRestartHint": "저장됨. 적용하려면 서버 재시작 (중지 → 시작).",
  "config.saved": "저장됨.",
  // savedToast 옆 인라인 액션 버튼 — 헤더의 토글을 찾지 않아도
  // 한 번에 "중지 → 대기 → 시작" 까지 끝내는 단축키.
  "config.restartNow": "지금 재시작",
  "config.restarting": "서버 재시작 중…",
  "config.restarted": "서버 재시작 완료.",

  // ── 진단 패널 ───────────────────────────────────────────────────
  "doctor.title": "진단",
  "doctor.run": "점검 실행",
  "doctor.running": "실행 중…",
  "doctor.empty": "리포트가 없습니다. 점검 실행을 눌러 시작하세요.",
  "doctor.overall.healthy": "정상",
  "doctor.overall.degraded": "주의",
  "doctor.overall.blocked": "차단",
  "doctor.overall.unknown": "알 수 없음",
  "doctor.status.pass": "통과",
  "doctor.status.warn": "주의",
  "doctor.status.fail": "실패",
  "doctor.fix": "해결",
  "doctor.fixing": "해결 중…",
  "doctor.fixHint": "권장 조치:",
  "doctor.fixCommand": "명령어:",
  "doctor.recheck": "다시 점검",
  "doctor.checking": "점검 중…",
  "doctor.intro":
    "진단은 앱 시작 시 그리고 수동 요청 시 실행됩니다. 각 항목은 해결 방법으로 연결됩니다.",
  "doctor.idle": "다시 점검을 눌러 진단을 실행하세요.",
  "doctor.working": "처리 중…",
  "doctor.fixIt": "자동 해결",
  "doctor.failed": "실패:",

  // ── 업데이트 패널 ───────────────────────────────────────────────
  "update.title": "Lumen 업데이트",
  "update.checking": "업데이트 확인 중…",
  "update.check": "업데이트 확인",
  "update.current": "현재 버전:",
  "update.latest": "최신 버전:",
  "update.upToDate": "최신 상태입니다.",
  "update.available": "업데이트 가능",
  "update.install": "업데이트 설치",
  "update.installing": "설치 중…",
  "update.serverWarn":
    "설치 전에 서버를 중지하세요 — 적용을 위해 앱이 재시작됩니다.",
  "update.releaseNotes": "릴리스 노트",
  "update.published": "게시일",
  "update.error": "업데이트 오류:",
  "update.installRestart": "설치 후 재시작",
  "update.applying": "— 적용 중…",
  "update.confirm.running":
    "추론 서버가 실행 중입니다. 업데이트를 설치하면 서버가 중지되고 앱이 재시작됩니다. 계속하시겠습니까?",
  "update.availableSuffix": "사용 가능",
  "update.onLatest": "최신 버전입니다.",

  // ── 진단 항목 이름 (백엔드 check.id 키) ─────────────────────────
  "doctor.check.os_version.name": "macOS 버전",
  "doctor.check.architecture.name": "CPU 아키텍처",
  "doctor.check.ram.name": "총 RAM",
  "doctor.check.disk_free.name": "여유 디스크 공간",
  "doctor.check.models_dir.name": "모델 디렉터리",
  "doctor.check.server_binary.name": "lumen-server 바이너리",
  "doctor.check.port_free.name": "서버 포트",
  "doctor.check.active_model.name": "활성 모델",
  "doctor.check.huggingface.name": "Hugging Face 네트워크",

  // ── 진단 메시지 템플릿 ──────────────────────────────────────────
  "doctor.msg.os_version.ok": "macOS",
  "doctor.msg.os_version.warn.suffix": "— 지원되지만 14 이상 권장",
  "doctor.msg.os_version.fail.suffix": "— 지원되지 않음",
  "doctor.msg.os_version.unknown": "macOS 버전 확인 불가",
  "doctor.msg.architecture.silicon": "Apple Silicon (arm64)",
  "doctor.msg.architecture.intel": "Intel Mac (x86_64)",
  "doctor.msg.architecture.other": "지원되지 않는 아키텍처:",
  "doctor.msg.ram.gb": "GB",
  "doctor.msg.disk_free.template": "{path} 에 {gb} GB 여유",
  "doctor.msg.models_dir.writable": "쓰기 가능:",
  "doctor.msg.models_dir.notWritable": "쓰기 불가:",
  "doctor.msg.models_dir.missing": "없음:",
  "doctor.msg.server_binary.found": "발견:",
  "doctor.msg.server_binary.notExecutable": "실행 권한 없음:",
  "doctor.msg.server_binary.notFound": "찾을 수 없음",
  "doctor.msg.port_free.available": "포트 {port} 사용 가능",
  "doctor.msg.port_free.inUse": "포트 {port} 사용 중",
  "doctor.msg.active_model.none": "선택된 모델 없음",
  "doctor.msg.active_model.ready": "{id} 준비 완료",
  "doctor.msg.active_model.incomplete": "{id} 디스크에 있으나 불완전",
  "doctor.msg.active_model.missing": "{id} 디스크에 없음",
  "doctor.msg.huggingface.reachable": "연결 가능 ({code})",
  "doctor.msg.huggingface.unexpected": "예상치 못한 상태 {code}",
  "doctor.msg.huggingface.unreachable": "연결 불가",
  "doctor.msg.huggingface.clientInitFailed": "HTTP 클라이언트 초기화 실패",

  // ── 진단 해결 안내 ──────────────────────────────────────────────
  "doctor.hint.os_version.warn":
    "macOS 14 (Sonoma)에서 Apple Silicon MPS 성능 개선이 적용되었습니다. 이전 버전도 동작하지만 약간의 성능 손실이 있습니다.",
  "doctor.hint.os_version.fail":
    "lumen은 Metal 스택 사용을 위해 macOS 11 (Big Sur) 이상이 필요합니다. 시스템 설정 → 일반 → 소프트웨어 업데이트에서 업그레이드하세요.",
  "doctor.hint.os_version.unknown":
    "`sw_vers` 실행 불가. macOS 환경에서 이상 동작입니다 — 이슈로 보고해주세요.",
  "doctor.hint.architecture.intel":
    "Intel Mac에서도 Metal이 동작하지만 Apple Silicon이 추론에 5–20배 빠르며 공식 개발 대상입니다. 더 작은 모델 (<2B)을 사용하거나 M 시리즈 기기에서 실행을 권장합니다.",
  "doctor.hint.architecture.unsupported":
    "lumen은 macOS Apple Silicon (및 제한적 Intel Mac) 대상입니다. 다른 플랫폼은 아직 지원되지 않습니다.",
  "doctor.hint.ram.warnLow":
    "16 GB는 1.5–7B 모델에 충분합니다. 13B 이상이나 Mixture-of-Experts 모델은 24 GB 이상 권장합니다.",
  "doctor.hint.ram.tight":
    "8–16 GB는 빠듯합니다. 2B 미만 파라미터 모델만 사용하세요. OOM 발생 시 3-bit 캐시 양자화를 켜고 wired 메모리 캡을 비활성화 (서버 카드 → 모든 캡 무시)하세요.",
  "doctor.hint.ram.fail":
    "RAM 8 GB 미만 — 거의 모든 모델에서 OOM이 발생합니다. RAM을 확보하거나 더 큰 머신을 사용하세요.",
  "doctor.hint.disk_free.warn":
    "최신 가중치 세트는 일반적으로 5–30 GB입니다. 여유 공간을 주시하세요 — 모델 카드에서 모델별 크기를 확인할 수 있습니다.",
  "doctor.hint.disk_free.fail":
    "여유 공간이 20 GB 미만입니다. 모델 다운로드가 중간에 실패합니다. 디스크를 확보하거나 서버 → 가중치 경로에서 위치를 변경하세요.",
  "doctor.hint.models_dir.notWritable":
    "모델 디렉터리의 권한을 수정하거나 서버 → 가중치 경로에서 변경하세요.",
  "doctor.hint.models_dir.missing": "자동 해결을 눌러 디렉터리를 생성하세요.",
  "doctor.hint.server_binary.notExecutable":
    "바이너리에 실행 비트를 설정하거나 다시 빌드하세요.",
  "doctor.hint.server_binary.notFound":
    "추론 서버를 소스에서 빌드하거나 config.toml의 SERVER → server_binary_path를 설정하세요.",
  "doctor.hint.port_free.inUse":
    "다른 프로세스가 이 포트를 점유 중입니다. 서버 카드에서 PORT를 변경하거나 해당 프로세스를 종료하세요.",
  "doctor.hint.active_model.none":
    "활성 모델 카드에서 모델을 선택하거나 모델 카드를 통해 HF Hub에서 다운로드하세요.",
  "doctor.hint.active_model.incomplete":
    "모델 카드에서 재다운로드하거나 제거 후 다시 추가하세요.",
  "doctor.hint.active_model.missing":
    "모델 카드에서 다운로드하거나 다른 활성 모델을 선택하세요.",
  "doctor.hint.huggingface.unexpected":
    "huggingface.co가 응답했지만 2xx/3xx가 아닙니다. 서비스가 저하되었을 수 있으며 다운로드가 실패할 수 있습니다.",
  "doctor.hint.huggingface.unreachable":
    "HF Hub를 통한 모델 다운로드가 실패합니다. 인터넷 연결, VPN, 프록시를 확인하세요.",

  // ── 진단 상세 ───────────────────────────────────────────────────
  "doctor.detail.active_model.incomplete":
    "필수 파일 (config.json + safetensors/gguf 샤드 1개 이상)이 누락되었습니다.",
};
