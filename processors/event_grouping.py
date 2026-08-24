"""
같은 Fed speaker의 연속 wire headline을 하나의 event로 묶는다.

목적:
- Reuters/Google 기사 중복 제거가 아니라,
- Newsquawk/MNI 등에서 한 인터뷰 중 연속 발생한 headline들을
  하나의 발언 event로 묶어 aggregation하기 위함.

기존 dedup.py를 대체하지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


DEFAULT_EVENT_WINDOW_MINUTES = 45


def _speaker(item: Mapping) -> str:
    return str(
        item.get("matched_member")
        or item.get("member")
        or item.get("speaker")
        or ""
    ).strip()


def _parse_datetime(value):
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def _event_id(speaker: str, dt: datetime, index: int) -> str:
    safe_speaker = "_".join(speaker.upper().split())
    return f"{safe_speaker}_{dt:%Y%m%d_%H%M}_{index:02d}"


def group_wire_events(
    items: Iterable[Mapping],
    *,
    window_minutes: int = DEFAULT_EVENT_WINDOW_MINUTES,
) -> list[dict]:
    """
    동일 speaker의 headline이 window_minutes 이내에 이어지면 같은 event로 묶는다.

    반환 event:
    {
        event_id,
        member,
        started_at,
        ended_at,
        headline_count,
        headlines,
        combined_text,
        source_type
    }
    """
    prepared = []

    for raw in items:
        speaker = _speaker(raw)
        dt = _parse_datetime(raw.get("published_at"))

        if not speaker or dt is None:
            continue

        prepared.append((dt, speaker, dict(raw)))

    prepared.sort(key=lambda x: (x[1].lower(), x[0]))

    events = []
    current = None
    window = timedelta(minutes=window_minutes)

    for dt, speaker, item in prepared:
        if (
            current is None
            or current["member"].lower() != speaker.lower()
            or dt - current["ended_at"] > window
        ):
            current = {
                "event_id": "",
                "member": speaker,
                "started_at": dt,
                "ended_at": dt,
                "headline_count": 0,
                "headlines": [],
                "combined_text": "",
                "source_type": "WIRE_EVENT",
            }
            events.append(current)

        text = str(
            item.get("article_text")
            or item.get("title")
            or ""
        ).strip()

        current["ended_at"] = dt
        current["headline_count"] += 1
        current["headlines"].append(item)

        if text:
            current["combined_text"] += (
                ("\n" if current["combined_text"] else "") + text
            )

    for index, event in enumerate(events, start=1):
        event["event_id"] = _event_id(
            event["member"],
            event["started_at"],
            index,
        )

    return events
