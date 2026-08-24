"""
기존 fed_speaker_monitor_v2에 붙일 때의 최소 사용 예.

이 파일 자체가 기존 pipeline을 교체하지 않는다.
현재 LLM 호출 함수만 `llm_call` 인자로 넘긴다.
"""

from __future__ import annotations

from fed_speaker_monitor_v2.collectors.article_text import extract_article_text
from fed_speaker_monitor_v2.llm.context import analyze_context
from fed_speaker_monitor_v2.validation.fomc_roberta import validate_direction
from fed_speaker_monitor_v2.validation.score_fusion import validate_score


def analyze_web_statement(
    *,
    url: str,
    speaker: str,
    fallback_text: str,
    llm_call,
) -> dict:
    text = extract_article_text(url, fallback_text)

    context = analyze_context(
        text=text,
        speaker=speaker,
        llm_call=llm_call,
    )

    if not context.relevant:
        return {
            "speaker": speaker,
            "url": url,
            "relevant": False,
            "score": 0.0,
            "summary": context.reason,
        }

    roberta = validate_direction(text)
    checked = validate_score(context.score, roberta)

    return {
        "speaker": speaker,
        "url": url,
        "relevant": True,
        "forward_looking": context.forward_looking,
        "score": checked["score"],
        "direction": checked["direction"],
        "review": checked["review"],
        "summary": context.reason,
    }
