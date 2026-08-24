from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fed_speaker_monitor_v2.collectors.fed_regional import collect_regional_official
from fed_speaker_monitor_v2.config import RESULTS_DIR
from fed_speaker_monitor_v2.models import Document


def _year_from_iso(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).year
    except ValueError:
        return None


def _attach_date_type(document, date_type):
    try:
        document.date_type = date_type
    except Exception:
        pass

    return document


def _row_to_document(row):
    """
    documents.json의 기존 regional row를 Document로 복원한다.
    Document 생성자에 실제로 쓰는 핵심 필드만 넘긴다.
    """
    document = Document(
        source=row.get("source", ""),
        title=row.get("title", ""),
        url=row.get("url", ""),
        published_at=row.get("published_at"),
        speaker=row.get("speaker"),
        text=row.get("text", ""),
        fetch_ok=bool(row.get("fetch_ok", False)),
    )

    return _attach_date_type(
        document,
        row.get("date_type"),
    )


def load_existing_regional_documents(
    target_year=2026,
    cache_path=None,
):
    """
    이전 pipeline 실행 결과 중 Regional Fed 문서만 읽는다.
    파일이 없거나 깨져 있으면 빈 리스트를 반환한다.
    """
    path = Path(
        cache_path
        or (RESULTS_DIR / "documents.json")
    )

    if not path.exists():
        return []

    try:
        rows = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return []

    documents = []

    for row in rows:
        source = str(
            row.get("source", "")
            or ""
        ).lower()

        if not source.startswith("regional_"):
            continue

        if _year_from_iso(
            row.get("published_at")
        ) != target_year:
            continue

        if not row.get("url"):
            continue

        if not row.get("text"):
            continue

        documents.append(
            _row_to_document(row)
        )

    return documents


def collect_regional_incremental(
    target_year=2026,
    cache_path=None,
):
    """
    1) 기존 documents.json의 Regional 문서를 읽고
    2) 기존 URL을 fed_regional.py에 skip_urls로 넘겨
    3) 신규 URL만 detail/body fetch한 뒤
    4) 기존 + 신규 Regional 문서를 반환한다.

    index/listing 페이지 확인은 계속 필요하다.
    대신 이미 저장된 speech detail/body 재수집을 피한다.
    """
    existing = load_existing_regional_documents(
        target_year=target_year,
        cache_path=cache_path,
    )

    existing_urls = {
        document.url
        for document in existing
        if getattr(document, "url", None)
    }

    print(
        f"[INCREMENTAL] cached regional documents: "
        f"{len(existing)}"
    )
    print(
        f"[INCREMENTAL] cached regional URLs: "
        f"{len(existing_urls)}"
    )

    new_documents = collect_regional_official(
        target_year=target_year,
        skip_urls=existing_urls,
    )

    print(
        f"[INCREMENTAL] new regional documents: "
        f"{len(new_documents)}"
    )

    merged = []
    seen_urls = set()

    for document in (
        existing
        + new_documents
    ):
        url = getattr(
            document,
            "url",
            None,
        )

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        merged.append(document)

    print(
        f"[INCREMENTAL] regional total: "
        f"{len(merged)}"
    )

    return merged