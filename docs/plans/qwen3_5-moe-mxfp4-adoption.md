# Full Adoption Plan: `mlx-community/Qwen3.6-35B-A3B-mxfp4`

## Context

사용자가 Qwen3.5-VL-MoE (35B total / 3B active, MXFP4 양자화, 멀티모달) 모델을 lumen-rs에 **풀 구현(멀티모달 포함)** 방향으로 도입하기로 결정. 본 문서는 Phase 1 CLAUDE.md 범위를 초과하므로 **새 로드맵(Phase 1.5)**을 추가 정의하고, 작업 단계·의존성·검증 기준을 정리한다.

본 작업은 코드 1~2만 LOC 규모이며 최소 3~4단계의 PR로 나눠야 한다. **단일 세션 구현 불가**, 단계별 체크인 기반으로 진행한다.

---

## 1. 대상 모델 핵심 스펙 (의사결정 근거)

| 항목 | 값 |
|---|---|
| `model_type` / arch | `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` |
| Layers / hidden | 40 / 2048 |
| Attention | 16 Q heads, 2 KV heads (GQA), head_dim 256, partial_rotary 0.25 |
| **Layer pattern** | linear×3 + full×1 × 10 cycles (30 linear + 10 full); full at idx 3,7,11,15,19,23,27,31,35,39 |
| **Linear attn** | **Mamba2 SSM** (conv_kernel=4, key_heads=16/val_heads=32, head_dim=128, float32 state) |
| **Full attn** | GQA + QK-norm + `attn_output_gate: true` (Qwen3-Next 계열) |
| MoE | 256 experts top-8 + shared_expert; `switch_mlp.*` 는 **단일 그룹 텐서** |
| Quantization | `mxfp4` (E2M1 + E8M0, group 32) 대부분; gate만 int8-affine group 64 (`.biases` 보유) |
| RoPE | `mrope_interleaved`, section=[11,11,10], θ=1e7, partial_rotary=0.25 |
| Vocab / max_pos | 248,320 / 262,144 |
| **Vision tower** | ViT 27-block, 1152-d, patch=16, merger(spatial 2×2, temporal 2) → 2048. **BF16 평문 (양자화 안 됨)** |
| Embed/Head | `embed_tokens`, `lm_head` 도 MXFP4 양자화됨 (주의) |
| 파일 | 4-shard safetensors 19.3GB + tokenizer 20MB, 총 1658 weight keys (text 1325 + vision 333) |

---

## 2. 현재 코드베이스 진입점 (이미 조사됨)

- **Candle**: 로컬 포크 `../candle/` (수정 가능)
- **구현된 모델**: Qwen2, Gemma3n/4 (dense, text-only)
- **MoE / MXFP4 / MLX / 비전**: 0건
- **TurboQuant 핵심**: 완성 ([crates/lumen-core/](../../crates/lumen-core/))
- **Metal 백엔드**: 완성 ([crates/lumen-metal/](../../crates/lumen-metal/))
- **attention.rs**: 13줄 스텁, 통합 지점
- **서버**: atomic_http 기반, OpenAI 호환 엔드포인트 동작 중

---

## 3. 단계별 구현 계획

### Stage 0 — 전제 확인 & 환경 (하루)
- **하드웨어 확인**: M-series Mac, **unified memory ≥ 64GB** 필수 (19.3GB 가중치 + KV + 활성화 + vision tower)
- `Cargo.toml` workspace에 신규 크레이트 선언 자리 확보
- upstream BF16 버전(`Qwen/Qwen3-VL-30B-A3B-Instruct` 또는 유사)을 레퍼런스로 확보 — 단위테스트용 소규모 레이어 추출 용도
- HF transformers `modeling_qwen3_5_moe.py` 참조 구현 스냅샷 확보 (라이선스 주의, 직접 복사 금지)

### Stage 1 — MXFP4 디코더 & Metal 커널 (최대 난이도, 선행)
**산출물**: MXFP4 텐서 로드 + matmul 동작

1. `crates/lumen-metal/src/mxfp4.rs` 신규
   - MXFP4 포맷: E2M1 4-bit element (sign + 2-bit exp + 1-bit mantissa, 8-value lookup) + **E8M0** (1-byte power-of-2 exponent) shared scale per 32-element group
   - 입력: `weight: [..., packed/8] u32`, `scales: [..., groups] u8(E8M0)` — MLX 저장 규약 (리틀엔디언 니블)
   - 디코드 LUT: `[±0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0]`
2. Metal 쉐이더 `mxfp4_matmul.metal`
   - **fused dequant+matmul** (weight-only, activation은 f16/bf16) — naive dequant-then-matmul 금지(메모리 피크 40GB)
   - int8-affine 게이트 전용 경로 별도: `w_real = (w_int8 - bias) * scale`, group_size 64
3. Candle 포크에 `MxFp4QTensor` variant 추가 — `quantized/mod.rs` 확장 또는 `QMatMul` trait 구현
4. Safetensors 로더: `.weight` / `.scales` / `.biases` 자동 페어링 (`.biases` 존재 시 int8-affine, 없으면 mxfp4로 판단)
5. **검증**: MLX Python에서 동일 레이어 dequant 결과 저장 → Rust 디퀀트 비트-정확 일치; 그 뒤 matmul L2 ≤ 1e-3

### Stage 2 — Qwen3.5-MoE 코어 아키텍처 (text path만)
**산출물**: 텍스트 입력 → 다음 토큰 로짓

1. `crates/lumen-model/src/qwen3_moe/mod.rs` 신규 모듈 (기존 `qwen.rs`는 Qwen2 전용으로 분리)
2. 구성요소:
   - `config.rs`: Qwen3_5MoeConfig struct (text_config/vision_config 중첩, quantization override 맵, layer_types 리스트)
   - `router.rs`: 256-expert top-8 softmax router + shared_expert_gate (sigmoid, 8-bit affine)
   - `expert.rs`: SwiGLU, **grouped-matmul expert** (leading dim=256, bmm 또는 gather→matmul)
   - `attention_full.rs`: GQA + QK-norm + partial_rotary + mRoPE section=[11,11,10] + `attn_output_gate`
   - `attention_linear.rs`: **Mamba2 SSM** — `in_proj_{a,b,qkv,z}` 결합, 1D causal conv(k=4), selective scan(f32 state), out_proj. state 캐시 구조 정의
   - `layer.rs`: `layer_types` 배열 따라 분기 (linear 30개 + full 10개)
   - `model.rs`: 40-layer forward + RMSNorm + **양자화된 embed_tokens/lm_head dequant** (tie=false)
3. `loader.rs` 확장:
   - `model.safetensors.index.json` 파싱 → 4-shard mmap
   - 접두사 `language_model.model.*` 매핑
   - `switch_mlp.{gate,up,down}_proj`는 그룹 텐서 1개로 로드 (per-expert 분할 금지)
   - `.scales` / `.biases` 동반 텐서 자동 페어링 → `QTensor` 래핑
4. **검증**: 단일 프롬프트 "The capital of France is " → "Paris" 생성, HF와 KL divergence ≤ 0.1

### Stage 3 — 토크나이저 & 챗 템플릿
1. `tokenizers` crate로 `tokenizer.json` (vocab 248K) 로드 — 기존 인프라 재사용
2. `minijinja` crate 추가 → `chat_template.jinja` 파싱
3. Qwen3 특수 토큰 매핑 (thinking blocks `<think>` 등), 새 EOS ID
4. `crates/lumen-server/src/routes/chat.rs` 연결
5. **검증**: 멀티턴 대화, 시스템 프롬프트 적용 일치

### Stage 4 — 멀티모달 파이프라인 (이미지 + 비디오)
1. `preprocessor_config.json` 파싱 → 이미지 resize/normalize (Qwen2-VL smart resize 규칙 재사용)
2. `video_preprocessor_config.json` → 비디오 프레임 샘플링 + temporal_patch=2 그룹핑
3. Vision tower 모듈 (`vision_tower.*` 접두사, ViT 27-block, **BF16 평문** — 양자화 경로 불필요)
   - `patch_embed.proj` (Conv3d or Conv2d stack)
   - `pos_embed` (learnable, 2304)
   - blocks: self_attn(qkv bias 있음) + MLP(linear_fc1/fc2) + 2×LayerNorm(bias 있음)
   - `merger.linear_fc1/fc2` + norm: spatial 2×2 + temporal 2 토큰 병합 → 2048-d 언어 임베딩 차원
4. image_token_id=248056 / video_token_id=248057 플레이스홀더 위치에 vision embedding 삽입 → language model 입력 임베딩 병합
5. OpenAI 호환 이미지 입력 API (`content: [{type: "image_url", ...}]`) 추가
6. **검증**: 이미지 URL 입력 → 설명 생성, 비디오 입력 → 요약

### Stage 5 — TurboQuant KV 캐시 통합
1. `attention.rs` 스텁 완성 — full attention 10개 레이어에 TurboQuant 훅
2. Linear attention 레이어는 recurrent state이므로 TurboQuant 미적용 (또는 별도 state 압축 전략)
3. `turboquant-cache/simple.rs`를 Qwen3.5-MoE 레이아웃에 맞춰 확장 (full attention layer indices만 캐시)
4. **검증**: FP16 KV 대비 메모리 ≤ 25% (3-bit), 생성 텍스트 일관성

### Stage 6 — 엔드투엔드 & 서버
1. `InferenceEngine` 아키텍처 디스패치에 `qwen3_5_moe` 추가 ([engine.rs](../../crates/lumen-server/src/engine.rs))
2. `CLAUDE.md` Phase 1 정의 업데이트 (타겟 모델 변경)
3. 부하 테스트: 토큰당 레이턴시 p95, 256K 컨텍스트에서 KV 압축률
4. **검증**: `curl POST /v1/chat/completions` 텍스트+이미지 요청 정상 응답

---

## 4. 중요 파일 (진입점)

**신규 생성**
- `crates/lumen-metal/src/mxfp4.rs` + `mxfp4_matmul.metal`
- `crates/lumen-model/src/qwen3_moe/` (모듈 전체)
- `crates/lumen-model/src/vision/` (vision tower)
- `crates/lumen-model/src/preprocessor.rs`

**수정**
- [crates/lumen-model/src/loader.rs](../../crates/lumen-model/src/loader.rs) — 4-shard index 파서, MXFP4 scales 필드
- [crates/lumen-model/src/config.rs](../../crates/lumen-model/src/config.rs) — Qwen3_5MoeConfig
- [crates/lumen-model/src/attention.rs](../../crates/lumen-model/src/attention.rs) — TurboQuant 훅 활성화
- [crates/lumen-server/src/engine.rs](../../crates/lumen-server/src/engine.rs) — 아키 디스패치
- [crates/lumen-server/src/routes/chat.rs](../../crates/lumen-server/src/routes/chat.rs) — 멀티모달 콘텐츠
- [crates/lumen-server/src/types.rs](../../crates/lumen-server/src/types.rs) — image/video content block
- [CLAUDE.md](../../CLAUDE.md) — Phase 1 타겟 재정의
- `Cargo.toml` (workspace + 각 crate) — `minijinja`, `image`, `video-rs` 또는 `ffmpeg-next` 의존성

**Candle 포크** (`../candle/`)
- `candle-core/src/quantized/` — MXFP4 QTensor variant
- `candle-nn` — 필요시 MoE 유틸 추가

---

## 5. 재사용 가능한 기존 자산

- `lumen-core` (lloyd_max/rotation/qjl) — 완전 재사용
- `lumen-metal` buffer/pipeline/device — MXFP4 커널 추가 시 인프라 재사용
- `lumen-server` HTTP/OpenAI 타입 — 멀티모달 content block만 추가
- `tokenizers` + `hf-hub` — 버전 그대로
- ~~`paged-attention` crate — Stage 5 이후 full attention 레이어에 연결 검토~~
  → 크레이트 삭제됨(2026-08-11). 측정 결과 되찾을 메모리가 프로세스의 1% 미만.
  `docs/maintainer-workflow.md` §9 참조.

---

## 6. 리스크 조사 결과 (2026-04-20, 결론)

`config.json` + `model.safetensors.index.json` (1658 keys) 전수 분석으로 아래 사실 확정:

### 6.1 Linear attention = **Mamba2 SSM** (확정)
- config: `mamba_ssm_dtype: float32`, `linear_conv_kernel_dim: 4`, `linear_num_key_heads: 16`, `linear_num_value_heads: 32`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128`
- 가중치 키 (layer 0 기준):
  ```
  linear_attn.A_log              # SSM decay params (log-domain)
  linear_attn.conv1d.weight      # 1D causal conv, kernel=4
  linear_attn.dt_bias            # discretization time-step bias
  linear_attn.in_proj_a.{weight,scales}    # Mamba2 4-way split input proj
  linear_attn.in_proj_b.{weight,scales}
  linear_attn.in_proj_qkv.{weight,scales}
  linear_attn.in_proj_z.{weight,scales}    # gate z
  linear_attn.norm.weight        # post-SSM RMSNorm
  linear_attn.out_proj.{weight,scales}
  ```
- **구현 레퍼런스**: Mamba2 (Tri Dao, 2024) / Qwen3-Next (공개된 Mamba2 블록 구조)
- 파급: `attention_linear.rs`는 **Mamba2 selective state space model** 구현 — SSM recurrence + 1D conv + gating. RWKV/DeltaNet 아님

### 6.2 Full attention = **GQA + QK-norm + output gate** (확정)
- layers 3/7/11/15/19/23/27/31/35/39 (10개)
- 키: `self_attn.{q,k,v,o}_proj.{weight,scales}` + `self_attn.{q,k}_norm.weight`
- `attention_bias: false` (proj에 bias 없음), `attn_output_gate: true` (Qwen3-Next 특성)
- 16 Q heads, 2 KV heads, head_dim 256, partial_rotary_factor 0.25 + mRoPE

### 6.3 MoE = 그룹 텐서 방식 (확정)
- `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales}` — **단일 그룹 텐서** (expert dim이 leading axis, per-expert 분할 아님). 로더에서 per-expert 반복 언패킹 불필요
- `mlp.shared_expert.{gate,up,down}_proj.{weight,scales}` — always-active
- `mlp.gate.{weight,scales,biases}` — 256-way router (8-bit 어파인, biases = zero-point)
- `mlp.shared_expert_gate.{weight,scales,biases}` — shared expert 시그모이드 게이트 (8-bit)

### 6.4 Vision Tower **존재 확정** (config·체크포인트 모두)
- 333개 키, **BF16 평문** (scales 없음 → 양자화 안 됨)
- 구조:
  ```
  vision_tower.patch_embed.proj.{weight,bias}     # patch=16, 3ch → 1152
  vision_tower.pos_embed.weight                   # learnable, 2304 positions
  vision_tower.blocks.{0..26}.attn.qkv.{weight,bias}
  vision_tower.blocks.{0..26}.attn.proj.{weight,bias}
  vision_tower.blocks.{0..26}.mlp.linear_fc1.{weight,bias}  # 1152 → 4304
  vision_tower.blocks.{0..26}.mlp.linear_fc2.{weight,bias}
  vision_tower.blocks.{0..26}.norm1.{weight,bias}
  vision_tower.blocks.{0..26}.norm2.{weight,bias}
  vision_tower.merger.linear_fc1.{weight,bias}    # 1152 → ? (temporal merge)
  vision_tower.merger.linear_fc2.{weight,bias}    # → 2048 (out_hidden_size)
  vision_tower.merger.norm.{weight,bias}
  ```
- spatial_merge_size=2, temporal_patch_size=2 → merger가 2×2 spatial + 2 temporal 토큰 병합
- Qwen2-VL/Qwen3-VL과 동일 계열 ViT

### 6.5 MTP (Multi-Token Prediction) 제외 가능
- config `mtp_num_hidden_layers: 1` 있으나 **체크포인트에 MTP 가중치 0개**. 초기 구현에서 안전하게 스킵

### 6.6 MXFP4 저장 포맷 (MLX 규약)
- `.weight`: 4-bit E2M1 원소 8개를 `uint32`로 팩 (리틀엔디언 니블). shape은 `[..., out_dim/8]` 수준
- `.scales`: 그룹당 하나의 스케일 (group_size=32 → `[..., out_dim/32]`). **E8M0** (1-byte 지수) 저장
- 순수 MXFP4는 `.biases` 없음 (대칭 양자화)
- 8-bit 게이트 경로는 `.weight` + `.scales` + `.biases` 3개 (어파인 int8 with zero-point, group_size=64)
- 디퀀트 공식:
  - MXFP4: `w_real = e2m1_table[nibble] * 2^(scale_E8M0 - 127)`
  - int8 affine: `w_real = (w_int8 - bias) * scale`

### 6.7 추가 확정 사항
- **최상위 키 prefix**: `language_model.model.*` (loader에서 접두어 매핑 규칙 필요)
- **embed_tokens·lm_head도 MXFP4 양자화** (`.weight + .scales`) — 일반 Candle Embedding/Linear가 아닌 양자화 dequant 임베딩 필요
- **`model.norm.weight`만 평문 BF16** (최종 RMSNorm)
- EOS 토큰: `[248046, 248044]`, image_token 248056, video_token 248057, vision_start 248053, vision_end 248054
- 라이선스: **Apache 2.0** (모델카드 명시)

### 6.8 남은 미해결
- `attn_output_gate: true`의 실제 구현: 별도 게이트 proj이 없는 것으로 보아 Qwen3-Next처럼 `o_proj` 앞에 sigmoid(q) × v 형태 추정. 레퍼런스(HF transformers `modeling_qwen3_next.py` 또는 `modeling_qwen3_5_moe.py`) 소스 검증 필요
- 메모리 피크: 가중치는 이미 MXFP4(19.3GB)이므로 **fused dequant+matmul 커널** 사용 시 40GB 피크 발생 안 함. naive dequant-then-matmul 경로는 피할 것
- MLX `mx.quantize` 내부 구현이 C++ 바이너리에만 있어, 검증은 레퍼런스 텐서 하나 dequant해서 Python MLX 결과와 바이너리 일치 비교로 수행

---

## 7. 엔드투엔드 검증

- `cargo test -p lumen-core` — 회귀 없음
- `cargo test -p lumen-metal mxfp4` — MXFP4 matmul HF 레퍼런스 대비 L2 ≤ 1e-2
- Stage 2 완료 시: "Paris" 생성 + HF KL divergence ≤ 0.1
- Stage 4 완료 시: 이미지 입력 캡션 생성
- Stage 5 완료 시: KV 메모리 ≤ FP16의 25% (3-bit)
- Stage 6 완료 시: `curl POST /v1/chat/completions` 텍스트+이미지 정상 응답, 256K 컨텍스트 성공
- Smoke: `just test` 전 크레이트 통과, `cargo clippy --workspace` clean

---

## 8. 예상 공수 (참고치)

| Stage | 복잡도 | 예상 LOC |
|---|---|---|
| 1 MXFP4 | 매우 높음 (커널 + Candle 포크) | ~2,500 |
| 2 Qwen3.5-MoE core | 높음 | ~3,500 |
| 3 토크나이저/챗 | 중간 | ~400 |
| 4 멀티모달 | 매우 높음 (vision + video) | ~4,000 |
| 5 TurboQuant 통합 | 중간 | ~800 |
| 6 서버/E2E | 낮음 | ~600 |
| **합계** | | **~12,000 LOC** |

단일 세션 불가. **Stage 1부터 순차 PR 체크인** 권장.
