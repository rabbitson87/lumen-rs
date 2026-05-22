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
  "context.turboquant.on": "TurboQuant:",
  "context.turboquant.off": "· 기본 KV 메모리 (압축 없음)",
  "context.recommended": "이 Mac 권장 최대",
  "context.recommended.suffix": "토큰",
  "context.warn.turnOnTurboquant":
    "— 더 긴 컨텍스트는 TurboQuant를 켜야 안전합니다",

  // ── 양자화 (튜닝 탭) ────────────────────────────────────────────
  "quant.title": "QUANT",
  "quant.titleHint": "(TurboQuant KV 캐시)",
  "quant.master": "TurboQuant",
  "quant.mode": "TurboQuant 모드",
  "quant.mode.off": "끔",
  "quant.mode.on": "켬",
  "quant.mode.auto": "자동",
  "quant.autoThreshold": "자동 임계값 (토큰)",
  "quant.qjl": "QJL 잔차 (Stage 2)",
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

  // ── QUANT 카드 툴팁 ─────────────────────────────────────────────
  "quant.tooltip.master":
    "KV 캐시 양자화의 마스터 스위치 (Lloyd-Max + Haar 회전, Stage 1). 켜면 정확도 손실이 적은 대신 KV 메모리를 약 4–8× 절감합니다. 끄면 KV가 bf16으로 유지됩니다 — 긴 컨텍스트에서 품질 문제가 보이는 경우에만 권장합니다.",
  "quant.tooltip.mode":
    "끔: KV를 절대 압축하지 않음 (짧은 prompt 디코드가 가장 빠름). 켬: 항상 압축 (긴 컨텍스트 메모리 절감 최대, 디코드 약 20–56% 느림). 자동: 이번 요청의 prompt 가 아래 임계값 이상일 때만 압축 — 짧은 대화는 빠르게, 긴 컨텍스트는 메모리 절감. 요청별 결정은 `[gemma4] tq_auto: ...` 로 로그에 기록됩니다.",
  "quant.tooltip.autoThreshold":
    "자동 모드가 TurboQuant 를 켜는 prompt 토큰 수. 기본 4096 — 이하에서는 bf16 슬라이딩 캐시가 충분히 작아서 풀 속도 디코드가 이득이고, 이상에서는 TQ 의 step 당 오버헤드가 KV 대역폭 절감으로 상쇄됩니다.",
  "quant.tooltip.qjl":
    "Stage 1 잔차에 대한 비편향 1비트 보정. (원본 − 복원)을 m차원 가우시안 공간에 투영하고 부호만 패킹합니다. 약간의 추가 비용(K/V 벡터당 m/8 바이트, Gemma 4 슬라이딩 윈도우 m=1024 기준 약 25 MB)으로 Top-5 약 2–3% / 코사인 +0.003을 회복합니다. Stage 1이 켜져 있어야 합니다.",
  "quant.tooltip.bits":
    "KV 채널당 Lloyd-Max 비트 수. 4: 최고 품질, FP16 대비 약 4× 축소. 3: 균형 — 권장 기본값. 2: 최대 압축 (약 8× 축소), 품질 소폭 저하. Gemma 4의 슬라이딩 윈도우 KV에 적용됩니다.",

  // ── CONTEXT 카드 설명 ───────────────────────────────────────────
  "context.hint.max.prefix":
    "최대 시퀀스 길이 (토큰). 호스트 RAM이 모델의 네이티브 한계를 감당할 수 없을 때 모델의 max_position_embeddings를 제한합니다 (Gemma 4는 128K를 표방).",
  "context.hint.max.tqOn":
    "현재 TurboQuant 설정으로 대략 위에 표시된 KV 압축률 적용",
  "context.hint.max.tqOnRealistic": "— 이 Mac에서의 실질 한계:",
  "context.hint.max.tqOff":
    "TurboQuant 꺼짐 — KV는 bf16 유지, 이 Mac에서의 실질 한계는",
  "context.hint.max.tqOffRealistic": "현재",
  "context.hint.max.tqOffFallback": "모델 네이티브 최대치보다 훨씬 낮습니다",
  "context.hint.max.env": "환경 변수:",
  "context.hint.sliding":
    "슬라이딩 윈도우 어텐션 크기. 일부 레이어 (Gemma 4: 30개 중 25개)는 전체 시퀀스 대신 최근 N개 토큰에만 어텐션을 적용합니다 → 긴 컨텍스트에서 KV 메모리가 유한. 0 = 모델 내장 기본값 사용, N>0이면 오버라이드 (작을수록 KV 절감, 장거리 회상 약화).",
  "context.hint.sliding.stacks":
    "TurboQuant와 함께 적용됩니다 — 슬라이딩은 어떤 토큰을 유지할지 결정, TurboQuant는 어떻게 저장할지 결정.",
  "context.hint.prefill":
    "프롬프트 처리 청크 상한. 이 값보다 긴 프롬프트는 \"prompt too large\" 오류로 거부됩니다. 클수록 긴 프롬프트를 받지만 프리필 동안 피크 메모리도 증가 (어텐션 QK·T = 청크 × KV",

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

  // ── QUANT 비트 비교 설명 ────────────────────────────────────────
  "quant.hint.tqOff":
    "TurboQuant 꺼짐 — KV 캐시는 bf16 유지 (Gemma 4 11K 한국어 컨텍스트 기준 약 5 GB)",
  "quant.hint.smallerVsFp16": "× FP16 대비 축소",
  "quant.hint.cosine": "코사인",
  "quant.hint.top5": "Top-5",
  "quant.hint.vs4bit": "4비트 기준 대비:",
  "quant.hint.kvMemory": "KV 메모리",
  "quant.hint.baseline": "기준 (최고 품질)",

  // ── CONTEXT 배너 ────────────────────────────────────────────────
  "context.banner.smallerThanBf16": "× bf16 대비 축소",
  "context.banner.kvCache": "KV 캐시 약 ",

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
    "lumen-server 하위 프로세스에 전달되는 원시 환경 변수 오버라이드. UI에 노출되지 않은 일회성 노브에 유용합니다 (예: LUMEN_GEMMA4_FUSE_EXPERTS, LUMEN_AFFINE4_FORCE_CPU). 타입화된 UI 필드와 동일한 키는 강조 표시됩니다.",
  "env.row.remove": "제거",
  "env.row.shadowsPrefix": "⚠ 다음 UI 필드를 가립니다:",
  "env.add2": "+ 추가",
  "env.revert": "되돌리기",
  "env.applyHint": "변경 사항은 다음 서버 시작 시 적용됩니다.",
  "env.empty2": "설정된 오버라이드가 없습니다.",

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
    "8–16 GB는 빠듯합니다. 2B 미만 파라미터 모델만 사용하세요. OOM 발생 시 3비트 TurboQuant를 사용하고 wired 메모리 캡을 비활성화 (서버 카드 → 모든 캡 무시)하세요.",
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
