"""
선택적 FOMC-RoBERTa 방향 검증기.

모델:
    gtfintechlab/FOMC-RoBERTa

공식 model card label:
    LABEL_0 = Dovish
    LABEL_1 = Hawkish
    LABEL_2 = Neutral

주의:
현재 Hugging Face에서 모델 파일 접근 시 이용조건 동의/로그인이
필요할 수 있다. 따라서 이 모듈은 optional이다.
"""

from __future__ import annotations

from functools import lru_cache


MODEL_NAME = "gtfintechlab/FOMC-RoBERTa"

LABEL_MAP = {
    "LABEL_0": "DOVISH",
    "LABEL_1": "HAWKISH",
    "LABEL_2": "NEUTRAL",
}


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
    )


def validate_direction(text: str) -> dict | None:
    """
    사용 가능하면:
        {"direction": "HAWKISH", "confidence": 0.91}

    모델 다운로드/인증/의존성 문제면:
        None

    validator 실패 때문에 전체 pipeline이 멈추지 않도록 설계한다.
    """
    if not text or not text.strip():
        return None

    try:
        result = _pipeline()(text.strip())[0]
    except Exception:
        return None

    label = LABEL_MAP.get(str(result.get("label", "")), "NEUTRAL")
    return {
        "direction": label,
        "confidence": float(result.get("score", 0.0)),
    }
