"""Multilingual calibration corpus for Gemma 4 26B-A4B imatrix capture.

Coverage rationale (the failure modes we observed at 3-bit):
  - Long Korean context (11K Moltis prompts → "비비-비비" loop)
  - Multilingual mid-conversation drift (Korean → Turkish)
  - Chat templates (production wraps every request in <start_of_turn>/<end_of_turn>)
  - Code (LumenRS users include source in prompts)
  - English instructional (baseline comparison)

We don't need a huge corpus — activation magnitude statistics converge fast
(within ~5K-10K tokens for stable per-channel estimates). The goal is
DISTRIBUTIONAL coverage, not volume.

Each entry is a list[dict] with role/content matching Gemma 4 chat template.
The tokenizer.apply_chat_template will produce the realistic input shape.
"""

# fmt: off
KO_LONG_BLOCKS = [
    # Long-form Korean essays — typical paragraph length / vocabulary
    """현대 사회에서 인공지능은 단순한 기술적 도구를 넘어 우리의 일상과 사고방식 자체를 변화시키고 있다. 1950년대 앨런 튜링이 던진 "기계가 생각할 수 있는가"라는 질문 이후 70여 년이 지난 지금, 우리는 매일같이 대규모 언어 모델과 대화하며 그 답을 직접 마주하고 있다. 그러나 이 답은 결코 단순하지 않다. 모델이 생성하는 텍스트는 표면적으로는 인간의 언어와 구별이 어려울 정도로 자연스럽지만, 그 내부 메커니즘은 인간의 사고와 본질적으로 다르다. 인간이 추론할 때 동원하는 상식, 직관, 감정, 도덕적 판단은 모델에게는 통계적 패턴의 흔적으로만 존재한다. 그렇다면 우리가 인공지능과 협업할 때 가장 경계해야 할 점은 무엇인가. 가장 중요한 것은 '그럴듯해 보이는 답'을 '정확한 답'으로 착각하지 않는 것이다.""",
    """한국어는 교착어로서 어간에 다양한 조사와 어미가 결합하여 의미를 형성한다. 이러한 형태론적 특성은 자연어 처리 모델에게 영어와는 다른 도전 과제를 제공한다. 예를 들어 동사 '먹다'는 '먹었다', '먹었을까', '먹었었더라도' 등 수많은 활용형으로 변형되며, 각 활용형은 시제, 양태, 양보 등 복합적인 문법 정보를 담는다. 영어가 주로 어순과 분리된 조동사로 이러한 의미를 표현하는 반면, 한국어는 모든 정보를 동사구 내부에 압축한다. 이로 인해 한국어 BPE 토크나이저는 한 어절을 여러 토큰으로 분할하며, 모델은 토큰 경계를 가로지르는 형태소 정보를 학습해야 한다. 이러한 구조적 차이는 양자화 시 활성 분포에도 영향을 미친다.""",
    """양자화는 신경망 가중치를 더 적은 비트수로 표현하여 메모리 사용량과 추론 속도를 개선하는 기법이다. FP16에서 INT4로 양자화하면 모델 크기는 4분의 1이 되지만, 모든 가중치를 동일하게 양자화하면 출력 품질이 급격히 떨어진다. 특히 임베딩 레이어와 라우터 같은 민감 부위는 양자화 오류가 모든 후속 레이어로 전파되어 catastrophic한 결과를 초래한다. 이를 해결하기 위해 mixed-precision quantization이 사용되며, 중요한 레이어는 높은 정밀도로 유지하고 덜 중요한 레이어만 공격적으로 압축한다. 어떤 레이어가 중요한지는 정적 휴리스틱이 아닌 실제 활성 통계로 판단해야 한다. 이것이 imatrix 기반 양자화의 핵심 아이디어이다.""",
    """서울은 한반도의 중심에 위치한 대한민국의 수도이자 최대 도시이다. 면적 605제곱킬로미터에 약 950만 명의 인구가 거주하며, 수도권 전체로 확장하면 2500만 명 이상이 모여 사는 거대 도시권을 형성한다. 한강이 도시를 동서로 가로지르며 강북과 강남으로 나뉜다. 조선왕조 600년 동안 한양이라는 이름으로 수도 역할을 했으며, 1948년 대한민국 정부 수립 이후 공식적으로 서울로 명명되었다. 현재 서울은 정치, 경제, 문화의 중심지이며 글로벌 도시 순위에서 상위권을 차지한다. 1988년 올림픽과 2002년 월드컵을 성공적으로 개최했고, 한류 문화의 발신지로서 전 세계적 영향력을 확대하고 있다.""",
    # --- corpus extension v2 (Option C: bigger calibration for AWQ proxy) ---
    """삼국시대는 고구려, 백제, 신라 세 나라가 한반도와 만주 지역에서 정립했던 약 700년에 걸친 시대이다. 기원전 1세기경부터 7세기 후반까지 이어진 이 시기는 한국 고대사의 형성기로, 각 나라가 독자적인 정치 체제와 문화를 발전시켰다. 고구려는 광활한 영토와 강력한 군사력으로 동북아시아의 패권을 다투었으며, 광개토대왕과 장수왕 시기에 전성기를 이루었다. 백제는 해상 무역과 외교에 능했고, 일본 열도에 한자와 불교, 기술을 전파하여 동아시아 문화 교류의 가교 역할을 했다. 신라는 화랑 제도와 골품제로 사회를 조직했으며, 김유신과 김춘추의 활약으로 7세기 중엽 삼국 통일의 기반을 마련했다. 668년 신라가 당나라와 연합하여 고구려를 멸망시킨 후 당과의 전쟁을 통해 자주적 통일을 이루어 통일신라 시대가 시작되었다.""",
    """한반도의 기후는 사계절이 뚜렷한 온대 몬순 기후로, 위도와 지형에 따라 다양한 양상을 보인다. 봄은 3월부터 5월까지로 황사와 미세먼지가 잦지만 벚꽃과 진달래가 만개하며 농사가 시작된다. 여름은 6월에서 9월까지로 장마전선이 한반도를 관통하면서 연 강수량의 60퍼센트 이상이 집중적으로 내리며, 평균 기온은 25도 안팎이지만 최근 폭염일수가 빠르게 증가하고 있다. 가을은 9월 말부터 11월까지로 청명한 하늘과 단풍이 이어지며 일교차가 크다. 겨울은 12월에서 2월까지로 시베리아 고기압의 영향을 받아 삼한사온 현상이 나타나고, 영서·산간 지역은 영하 20도까지 떨어지는 일이 흔하다. 기후 변화로 인해 이러한 전통적 사계절 패턴이 약해지고, 여름과 겨울이 길어지며 봄·가을이 짧아지는 경향이 관측된다.""",
    """한국의 의료 시스템은 전 국민 건강보험 제도를 기반으로 한 보편적 의료 보장 체계이다. 국민건강보험공단이 단일 보험자로서 전 국민의 보험료를 통합 관리하며, 직장 가입자와 지역 가입자로 구분된다. 의료 기관은 1차 의원, 2차 병원, 3차 종합병원으로 단계별로 구성되어 환자가 증상의 심각도에 따라 적절한 수준의 의료 서비스를 받도록 설계되어 있다. 본인 부담률은 의료 기관 종류와 진료 내용에 따라 20에서 60퍼센트까지 차등 적용되며, 중증 질환과 희귀 난치성 질환은 산정 특례 제도를 통해 본인 부담을 크게 낮춘다. 한국 의료의 강점은 비교적 낮은 본인 부담, 짧은 대기 시간, 첨단 영상 진단 장비의 광범위한 보급에 있다. 다만 일차 의료 강화, 의료 인력의 지역 불균형, 만성 질환 관리의 체계화 등은 여전히 정책적 과제로 남아 있다.""",
    """현대 한국 문학은 일제 강점기와 한국전쟁이라는 역사적 격동 속에서 독자적 미학과 사회의식을 형성해왔다. 1920년대 이광수와 김동인이 도입한 근대 소설의 기법은 염상섭의 사실주의와 채만식의 풍자로 발전했으며, 윤동주와 정지용은 한국어 시의 음악성과 정신성을 깊이 있게 탐구했다. 분단 이후 황순원, 박경리, 박완서는 전쟁의 상흔과 가족 해체를 정밀하게 묘사했고, 1970년대 산업화 시기에는 조세희, 황석영, 이문열이 도시 빈민과 노동자의 삶을 문학의 중심으로 끌어올렸다. 한강이 2024년 노벨문학상을 수상하면서 한국문학은 세계문학사에 새로운 좌표를 새겼다. 한강의 작품은 광주민주화운동, 제주 4·3 사건, 여성 신체의 폭력성과 같이 한국사의 트라우마를 시적이면서도 단단한 산문으로 풀어낸다는 점에서 평론가들의 주목을 받았다.""",
    """블록체인은 분산 네트워크상의 노드들이 합의 알고리즘을 통해 거래 내역을 검증하고 변경 불가능한 원장에 기록하는 기술이다. 2008년 사토시 나카모토라는 익명의 인물이 발표한 비트코인 백서에서 작업 증명 기반의 첫 구현이 제시되었으며, 이후 이더리움이 스마트 계약 개념을 도입하여 단순 화폐를 넘어 프로그래머블한 분산 컴퓨팅 플랫폼으로 발전시켰다. 합의 알고리즘은 작업 증명, 지분 증명, 위임된 지분 증명, 비잔틴 장애 허용 등으로 다양화되었고, 각 방식은 보안성, 처리량, 에너지 효율의 트레이드오프를 다르게 가져간다. 한국에서는 카카오의 클레이튼, 라인의 핀시아처럼 대기업이 자체 블록체인을 운영했고, 정부도 디지털 신원 인증과 기록 관리에 블록체인을 활용하는 시범 사업을 진행했다. 다만 가상자산 시장의 변동성과 투기성, 규제 공백, 자금세탁 우려는 여전히 해결해야 할 과제이다.""",
    """한국 영화 산업은 1990년대 후반 이후 세계 영화 시장에서 가장 빠르게 성장한 영역 중 하나이다. 1999년 강제규 감독의 쉬리가 한국형 블록버스터의 가능성을 열었고, 박찬욱 감독의 올드보이가 2004년 칸 영화제 심사위원 대상을 받으며 한국 영화의 작가성이 세계적으로 인정받기 시작했다. 봉준호 감독의 기생충은 2019년 칸 영화제 황금종려상과 2020년 아카데미 시상식 작품상을 동시에 거머쥐며 비영어권 영화로서 새로운 역사를 썼다. 넷플릭스가 제작한 오징어 게임은 2021년 공개 후 전 세계 94개국에서 시청률 1위를 기록했고, 한국 드라마와 K팝이 결합된 한류 문화가 글로벌 콘텐츠 산업의 주요 축으로 자리잡았다. 이러한 성공의 배경에는 다양한 장르 실험을 허용하는 제작 환경, 강력한 시각 연출, 사회 비평을 담은 서사 등 한국 콘텐츠 산업의 구조적 강점이 자리한다.""",
    """양자 컴퓨팅은 양자역학의 중첩과 얽힘 원리를 활용하여 고전 컴퓨터로는 다항 시간 안에 풀기 어려운 특정 문제를 효율적으로 해결하는 새로운 계산 패러다임이다. 큐비트는 0과 1이 동시에 존재하는 중첩 상태를 가지며, 여러 큐비트가 얽히면 지수적으로 큰 상태 공간을 한 번에 표현할 수 있다. 1994년 피터 쇼어가 발표한 양자 알고리즘은 큰 정수를 다항 시간에 소인수분해할 수 있음을 보였는데, 이는 RSA 등 기존 공개키 암호 체계의 안전성을 근본적으로 위협한다. 1996년 그로버 알고리즘은 비정렬 데이터베이스 검색을 제곱근 속도로 가속화한다. 구글, IBM, 한국과학기술연구원, 삼성종합기술원 등이 초전도, 이온 트랩, 광자 등 다양한 큐비트 구현 방식을 연구하고 있으며, 양자 우월성 시연과 오류 정정 기술이 핵심 과제이다.""",
]

KO_QA_PAIRS = [
    ("한국 전통 음식 비빔밥의 역사와 영양학적 특징을 자세히 설명해주세요.",
     "비빔밥은 한국의 대표적인 전통 음식 중 하나로, 그릇에 밥과 다양한 나물, 고기, 계란 등을 담고 고추장과 함께 비벼 먹는 요리입니다. 역사적으로는 조선시대의 골동반(骨董飯)에서 유래했다고 알려져 있으며, 음력 정월 대보름이나 제사 후 남은 음식을 한 그릇에 모아 비벼 먹던 풍습이 발전한 것으로 추정됩니다. 영양학적 측면에서 비빔밥은 탄수화물, 단백질, 비타민, 무기질을 골고루 섭취할 수 있는 균형 잡힌 한 끼 식사로 평가받습니다."),
    ("머신러닝에서 과적합(overfitting)을 방지하는 방법들을 설명해주세요.",
     "과적합은 모델이 훈련 데이터에 너무 잘 맞춰져서 새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다. 이를 방지하는 주요 기법으로는 첫째, 규제화(L1, L2 regularization)를 통해 가중치 크기를 제한하는 방법, 둘째, 드롭아웃(dropout)으로 학습 중 임의로 뉴런을 비활성화하는 방법, 셋째, 데이터 증강(data augmentation)으로 훈련 데이터의 다양성을 늘리는 방법, 넷째, 조기 종료(early stopping)로 검증 손실이 증가하기 시작할 때 학습을 중단하는 방법 등이 있습니다."),
    ("Rust 언어의 소유권(ownership) 시스템이 메모리 안전성을 어떻게 보장하는지 설명해주세요.",
     "Rust의 소유권 시스템은 컴파일 타임에 메모리 안전성을 보장하는 핵심 메커니즘입니다. 세 가지 규칙으로 동작합니다. 각 값은 하나의 소유자만 가지며, 소유자가 스코프를 벗어나면 값은 자동으로 해제됩니다. 값을 다른 변수에 할당하면 소유권이 이전되며 원래 변수는 더 이상 사용할 수 없습니다. 참조(borrowing)를 통해 소유권을 이전하지 않고 값에 접근할 수 있지만, 가변 참조는 동시에 하나만 존재할 수 있습니다. 이러한 규칙은 use-after-free, double-free, 데이터 레이스를 컴파일 단계에서 차단합니다."),
    ("Apple Silicon의 unified memory architecture와 일반적인 PC의 메모리 구조 차이점은 무엇인가요?",
     "Apple Silicon의 unified memory architecture(UMA)는 CPU와 GPU가 동일한 물리적 메모리 풀을 공유하는 구조입니다. 일반적인 PC에서는 CPU가 시스템 RAM을 사용하고 GPU는 별도의 VRAM을 사용하며, 두 메모리 간 데이터 이동에는 PCIe 버스를 통한 명시적 복사가 필요합니다. UMA에서는 이 복사 단계가 제거되어 CPU에서 처리한 데이터를 GPU가 즉시 접근할 수 있고, 그 반대도 마찬가지입니다. 머신러닝 추론에서는 이러한 특성이 큰 이점을 제공합니다. 모델 가중치를 메모리에 한 번만 로드하면 CPU 전처리와 GPU 행렬 연산 모두에서 동일한 데이터를 사용할 수 있어 메모리 효율성과 지연 시간 모두에서 유리합니다."),
]

EN_LONG_BLOCKS = [
    """Modern large language models exhibit emergent capabilities that scale unpredictably with parameter count. While the underlying transformer architecture has remained remarkably stable since 2017, the practical behavior of models trained on trillions of tokens often surprises even their creators. Phenomena like chain-of-thought reasoning, in-context learning, and instruction following appear at certain scale thresholds without explicit training for these behaviors. This unpredictability creates a fundamental challenge for deployment: we cannot fully characterize what a model will do at inference time, only what it tends to do across a finite evaluation set. Mixture-of-experts architectures further complicate this picture by routing different tokens through different specialized sub-networks, creating a combinatorial explosion of effective compute paths.""",
    """Quantization-aware training and post-training quantization represent two fundamentally different approaches to compressing neural networks. PTQ operates on a fully trained model, computing statistics from a small calibration set to determine how to map floating-point weights onto a smaller integer codebook. This is fast, requires no additional training, and works reasonably well for ~4-bit precision. Below that, error accumulates rapidly across layers, and naive PTQ often produces models that generate gibberish or get stuck in degenerate loops. The importance matrix approach addresses this by measuring per-channel activation magnitudes during the calibration pass and using these to drive non-uniform bit allocation: channels carrying high activation energy receive higher precision, while quiet channels are aggressively compressed.""",
    """The Apple Metal Performance Shaders framework provides a high-level API for general-purpose GPU computation on Apple Silicon. Unlike CUDA, which exposes a C-like kernel language with extensive control over memory hierarchy, Metal Shading Language abstracts away many low-level details in favor of a more graphics-oriented model. For machine learning workloads, this creates both opportunities and constraints. The unified memory architecture eliminates explicit host-device transfers, but the lack of fine-grained synchronization primitives makes implementing complex parallel algorithms harder than on NVIDIA hardware. Recent versions of MLX, Apple's machine learning framework, address this gap by providing primitives like quantized matmul, scaled dot-product attention, and fused activation kernels that match or exceed CUDA equivalents on similar TDP envelopes.""",
    # --- corpus extension v2 ---
    """Climate science distinguishes weather from climate by time scale: weather refers to atmospheric conditions over hours to days, while climate is the statistical distribution of weather over decades or longer. The Earth's climate system involves five interacting components: atmosphere, hydrosphere, cryosphere, lithosphere, and biosphere. Greenhouse gases including carbon dioxide, methane, nitrous oxide, and water vapor trap longwave radiation emitted from the Earth's surface, raising global mean surface temperature beyond what solar radiation alone would produce. Since the industrial revolution, atmospheric CO2 has risen from approximately 280 parts per million to over 420 parts per million, driven primarily by fossil fuel combustion and land use change. Climate models project that without aggressive emissions reductions, global mean temperature will rise 2 to 4 degrees Celsius above preindustrial levels by 2100, with cascading effects on sea level, ecosystems, agriculture, and human health. Mitigation requires both reducing emissions and removing existing atmospheric carbon through forestation, soil management, and engineered solutions.""",
    """Distributed systems design must address the fundamental tradeoff between consistency, availability, and partition tolerance, a result formalized in the CAP theorem. When a network partition occurs, a system can either remain available by accepting potentially inconsistent reads and writes, or maintain strict consistency at the cost of rejecting requests on the minority side. Real systems navigate this tradeoff through nuanced choices: leader election protocols like Raft and Paxos provide linearizable consistency with majority-quorum availability, while gossip protocols and CRDTs sacrifice strict consistency for eventual convergence and high availability. The actual production constraints often include latency budgets, geographic distribution, hardware failure rates, and economic cost per byte stored. Modern systems like Spanner, FoundationDB, and CockroachDB use atomic clocks, hybrid logical clocks, and external consensus to deliver strong consistency at planetary scale, while DynamoDB and Cassandra prefer tunable consistency for higher availability and lower latency.""",
    """The history of the industrial revolution is often told as a sequence of technological breakthroughs: the steam engine, the spinning jenny, the Bessemer process, and so on. But the deeper transformation was institutional and social. England in the late eighteenth century happened to combine several preconditions that, while not unique on their own, were rare in combination: secure property rights enforced by an independent judiciary, expanding mercantile finance, a partial labor market created by enclosure of common lands, abundant coal located near navigable water, and a culture that, after the Glorious Revolution of 1688, increasingly tolerated experimental natural philosophy. None of these factors caused the revolution by itself. Their conjunction produced an environment in which incremental innovations could compound, where the returns to a successful improvement in a textile mill or coal mine could be captured by the inventor and reinvested, and where the political system did not crush new wealth in favor of old privilege. Subsequent industrial transformations in Germany, the United States, Japan, and South Korea each rewrote these preconditions for their own historical moment, but the core dynamic of compounding innovation under institutional protection remained.""",
    """Modern compiler optimization rests on a hierarchy of intermediate representations and transformation passes. After parsing, the front end produces an abstract syntax tree which is lowered to a typed intermediate representation, often in static single assignment form. SSA makes data dependencies explicit by giving each variable exactly one definition site, dramatically simplifying analyses like constant propagation, dead code elimination, and global value numbering. Optimizations are scheduled in a pass pipeline where ordering matters: inlining typically runs early to expose intraprocedural opportunities, while loop transformations like vectorization, unrolling, and parallelization run later when the loop structure has stabilized. Modern compilers like LLVM, GCC, and the JVM JIT also perform speculative optimizations under runtime assumptions, falling back to slower paths when assumptions are violated. Profile-guided optimization further specializes hot paths using observed execution frequencies. The interaction between these layers means that small changes to source code or compiler flags can have surprisingly large performance consequences, which is why benchmarking and disassembly inspection remain essential skills for systems engineers.""",
    """Microbiomes are the collective genetic material of microorganisms inhabiting a particular environment, from human gut to ocean sediment to plant rhizosphere. The human gut microbiome alone contains trillions of bacteria from over a thousand species, with a combined genome roughly 150 times larger than the human genome. These organisms perform functions the host cannot, including synthesis of certain vitamins, fermentation of dietary fiber into short chain fatty acids, modulation of immune responses, and competitive exclusion of pathogens. Disruption of the gut microbiome through antibiotic use, dietary shifts, or chronic stress has been linked epidemiologically to inflammatory bowel disease, metabolic syndrome, autoimmune disorders, and even neurological conditions through the gut-brain axis. The development of high-throughput sequencing has transformed microbiome research from culture-based methods, which captured only a small fraction of microbial diversity, to metagenomic and metatranscriptomic approaches that can describe entire communities and their functional activity in situ.""",
]

CODE_BLOCKS = [
    '''use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Engine {
    backend: Arc<dyn Backend>,
    config: GenerationConfig,
}

impl Engine {
    pub fn new(backend: impl Backend + 'static, config: GenerationConfig) -> Self {
        Self {
            backend: Arc::new(backend),
            config,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, EngineError> {
        let tokens = self.backend.tokenize(prompt)?;
        let mut output = Vec::with_capacity(self.config.max_tokens);
        let mut cache = self.backend.create_cache();

        for _ in 0..self.config.max_tokens {
            let logits = self.backend.forward(&tokens, &mut cache).await?;
            let next_token = self.sample(&logits)?;
            if next_token == self.backend.eos_token() {
                break;
            }
            output.push(next_token);
        }

        self.backend.detokenize(&output)
    }
}''',
    '''import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)''',
]

EN_QA_PAIRS = [
    ("Explain how attention masks work in transformer decoders during autoregressive generation.",
     "In transformer decoders, the attention mask enforces causality during training and inference. The mask is a lower-triangular boolean matrix where position (i, j) is True if and only if j <= i, meaning each token can attend to itself and all previous tokens but never future ones. During training with teacher forcing, this allows the decoder to be trained in parallel across all positions while preserving the autoregressive property. During inference, when generating one token at a time, the mask naturally extends as new tokens are appended to the KV cache. The mask is typically applied by adding negative infinity (or a very large negative number) to the attention scores at masked positions before the softmax."),
    ("What is the difference between greedy decoding and nucleus sampling?",
     "Greedy decoding always selects the token with the highest probability at each step, producing deterministic but often repetitive output. Nucleus sampling, also called top-p sampling, instead selects from a dynamic subset of tokens whose cumulative probability exceeds a threshold p (commonly 0.9). This subset adapts to the distribution shape: when the model is confident, it might include only one or two tokens; when uncertain, it expands to capture the long tail. The selected tokens are then renormalized and sampled from. Combined with temperature scaling, nucleus sampling produces more diverse and natural outputs than greedy decoding while avoiding the random low-probability tokens that pure multinomial sampling can produce."),
]


def build_chat_corpus(tokenizer, max_tokens_per_seq: int = 2048) -> list[list[int]]:
    """Materialize the corpus as a list of token-id sequences.

    Builds:
      - Pure long-context KR/EN blocks (no chat template) — represents
        document-style input the model will see at long contexts.
      - Chat-templated KR/EN Q&A — represents production chat mode.
      - Code blocks — represents typical LumenRS user prompts.

    Each entry is truncated to `max_tokens_per_seq` to bound calibration cost.
    """
    sequences = []

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # Raw long-context blocks — repeat aggressively to push each sequence
    # toward the truncation cap (max_tokens_per_seq) since we want to exercise
    # the long-context regime that 3-bit Gemma 4 specifically failed at.
    for block in KO_LONG_BLOCKS:
        ids = encode(block * 8)
        sequences.append(ids[:max_tokens_per_seq])
    for block in EN_LONG_BLOCKS:
        ids = encode(block * 8)
        sequences.append(ids[:max_tokens_per_seq])

    # Chat-templated Q&A — realistic production shape.  Repeat the answer
    # so the templated sequence is dense, not all `<turn|>` markers.
    for q, a in KO_QA_PAIRS + EN_QA_PAIRS:
        msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": (a + " ") * 4},
        ]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
        sequences.append(ids[:max_tokens_per_seq])

    # Mixed long-context + chat (the exact failure-mode shape: long Korean
    # context preceding a question).
    for block in KO_LONG_BLOCKS:
        msgs = [
            {"role": "user", "content": (block * 3) + "\n\n위 내용을 다섯 문장으로 요약해주세요."},
            {"role": "assistant", "content": "요약하면 다음과 같습니다. 첫째, 본문은 다양한 주제를 다루며 각 주제별로 중요한 개념을 설명합니다. 둘째, 핵심 메시지는 비판적 사고와 정확한 정보 검증의 중요성입니다. 셋째, 한국어 처리에는 고유한 언어학적 특성이 반영되어야 합니다. 넷째, 양자화는 모델 효율성을 높이지만 품질 저하를 동반할 수 있어 신중한 설계가 필요합니다. 다섯째, 결론적으로 기술과 인간의 협업이 핵심입니다."},
        ]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
        sequences.append(ids[:max_tokens_per_seq])

    # Code blocks (no chat template — raw)
    for code in CODE_BLOCKS:
        ids = encode(code * 4)
        sequences.append(ids[:max_tokens_per_seq])

    return sequences
