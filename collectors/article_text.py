"""
뉴스/공식 웹페이지 본문 정제.

기존 collector를 대체하지 않는다.
collector가 URL을 얻은 뒤 본문 품질이 낮을 때 이 함수만 호출한다.

Trafilatura 2.x:
    fetch_url(url) -> HTML
    extract(html, ...) -> clean text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import trafilatura


@dataclass
class ExtractedArticle:
    url: str
    text: str
    title: str = ""
    date: str = ""
    author: str = ""


def extract_article_text(
    url: str,
    fallback_text: str = "",
    *,
    favor_precision: bool = True,
) -> str:
    """
    URL에서 기사/공식 문서 본문을 추출한다.

    실패 시 예외를 올리지 않고 fallback_text를 반환한다.
    뉴스 RSS description을 fallback_text로 넘기면 안전하다.
    """
    if not url:
        return fallback_text.strip()

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return fallback_text.strip()

        text = trafilatura.extract(
            downloaded,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=favor_precision,
        )
        return (text or fallback_text).strip()
    except Exception:
        return fallback_text.strip()
