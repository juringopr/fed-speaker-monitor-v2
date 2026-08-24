"""
LLM context score와 FOMC-RoBERTa 방향을 최소 규칙으로 결합한다.

RoBERTa는 점수를 새로 만드는 모델이 아니라 '방향 검증' 역할만 한다.
LLM과 일치하면 그대로 통과하고, 불일치하면 review flag만 세운다.
"""

from __future__ import annotations


def score_to_direction(score: float, neutral_band: float = 0.15) -> str:
    if score > neutral_band:
        return "HAWKISH"
    if score < -neutral_band:
        return "DOVISH"
    return "NEUTRAL"


def validate_score(
    llm_score: float,
    roberta_result: dict | None,
    *,
    neutral_band: float = 0.15,
) -> dict:
    llm_direction = score_to_direction(llm_score, neutral_band)

    if roberta_result is None:
        return {
            "score": float(llm_score),
            "direction": llm_direction,
            "review": False,
            "validator": "unavailable",
        }

    rb_direction = str(roberta_result.get("direction", "NEUTRAL")).upper()
    agrees = llm_direction == rb_direction

    return {
        "score": float(llm_score),
        "direction": llm_direction,
        "review": not agrees,
        "validator": rb_direction,
        "validator_confidence": float(
            roberta_result.get("confidence", 0.0)
        ),
    }
