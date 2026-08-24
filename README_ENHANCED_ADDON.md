# Fed Speaker Monitor v2 — Enhanced Context Add-on

기존 `fed_speaker_monitor_v2`의 파일은 수정하지 않고
아래 보조 모듈만 추가하는 패키지입니다.

## 추가 파일

```text
fed_speaker_monitor_v2/
├─ collectors/
│  └─ article_text.py
├─ llm/
│  └─ context.py
├─ validation/
│  ├─ fomc_roberta.py
│  └─ score_fusion.py
└─ enhanced_adapter.py
```

## 역할

### 1. article_text.py

RSS/Google/Fed collector가 URL을 찾은 다음
Trafilatura로 메뉴/광고 등을 제거하고 본문을 정제합니다.

기존 collector를 대체하지 않습니다.

```text
기존 RSS/Google/Fed collector
        ↓ URL
article_text.py
        ↓ clean text
기존 processors 또는 context.py
```

### 2. context.py

WorldCentralBanks 연구 아이디어를 최소화하여:

```text
Monetary-policy relevant?
        ↓
Hawkish / Dovish / Neutral
        ↓
Forward-looking?
        ↓
Intensity 0~1
        ↓
Score -1~+1
```

을 한 번에 JSON으로 반환하도록 설계했습니다.

특정 OpenAI/Anthropic SDK를 새로 넣지 않았습니다.
현재 프로젝트에서 쓰는 LLM 호출 함수를:

```python
def llm_call(prompt: str) -> str:
    ...
```

형태로 전달하면 됩니다.

즉 기존 인증/클라이언트를 중복 구축하지 않습니다.

### 3. fomc_roberta.py

`gtfintechlab/FOMC-RoBERTa`를 optional validator로 사용합니다.

```text
LLM score
        ↓
direction
        ↔
FOMC-RoBERTa direction
        ↓
같음    → pass
다름    → review=True
```

RoBERTa가 최종 점수를 덮어쓰지 않습니다.

현재 Hugging Face 모델 저장소는 파일 접근 전에
사용조건 동의/로그인이 필요할 수 있으므로,
모델을 사용할 수 없으면 자동으로 `None`을 반환합니다.

따라서 이 기능 때문에 기존 pipeline이 멈추지 않습니다.

### 4. score_fusion.py

복잡한 ensemble을 만들지 않습니다.

LLM과 RoBERTa가 다를 경우 최종 점수를 억지로 평균내지 않고
`review=True`만 표시합니다.

이 방식이 현재 단계에서는 가장 추적하기 쉽습니다.

## 설치

필수:

```bash
pip install trafilatura
```

RoBERTa validation까지 사용할 경우:

```bash
pip install transformers torch
```

Hugging Face에서 모델 이용조건 동의가 필요할 수 있습니다.

## 현재 프로젝트와 연결

기존:

```text
collectors
→ document.py
→ dedup.py
→ segments.py
→ stance.py
→ validator.py
→ aggregation
```

는 유지합니다.

추가 모듈은 다음처럼 점진적으로 사용할 수 있습니다.

```text
collectors
    ↓ URL
article_text.py
    ↓
segments.py
    ↓
context.py / 기존 stance.py
    ↓
fomc_roberta.py (optional)
    ↓
score_fusion.py
```

## 가장 먼저 적용할 부분

처음에는 `article_text.py` + `context.py`만 사용하는 것을 권장합니다.

FOMC-RoBERTa는 실제 결과가 어느 정도 쌓인 후
LLM과 불일치율을 확인하면서 validator로 켜면 됩니다.

## 기존 파일 수정 여부

이 ZIP에는 기존 파일 수정본이 없습니다.

추가 파일만 복사한 뒤 개별 테스트할 수 있습니다.
