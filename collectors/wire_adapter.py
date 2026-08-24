"""
Paid/news wire 공통 정규화 어댑터.

Newsquawk, MNI, LSEG 등 어떤 공급자를 쓰더라도
최종적으로 같은 dict 구조로 바꿔 기존 processors에 넘기는 목적이다.

외부 API 호출은 하지 않는다.
공급자별 collector가 받은 raw item을 normalize_wire_item()에 전달한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_wire_item(
    raw: Mapping[str, Any],
    *,
    provider: str,
    speaker: str = "",
    text_field: str = "headline",
    time_field: str = "published_at",
    url_field: str = "url",
    source_field: str = "source",
) -> dict:
    """
    공급자 raw item -> Fed Speaker Monitor 공통 item.

    필요한 최소 필드만 만든다.
    필드명이 다른 API는 인자로 매핑하면 된다.
    """
    headline = _text(raw.get(text_field))
    published_at = raw.get(time_field)

    return {
        "source_type": "WIRE",
        "provider": _text(provider).upper(),
        "source": _text(raw.get(source_field)) or _text(provider),
        "matched_member": _text(speaker or raw.get("speaker")),
        "title": headline,
        "article_text": headline,
        "published_at": published_at,
        "url": _text(raw.get(url_field)),
        "wire_id": _text(
            raw.get("id")
            or raw.get("story_id")
            or raw.get("headline_id")
        ),
        "raw": dict(raw),
    }


def is_wire_item(item: Mapping[str, Any]) -> bool:
    return _text(item.get("source_type")).upper() == "WIRE"
