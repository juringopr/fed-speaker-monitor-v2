from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fed_speaker_monitor_v2.aggregation.final import build_final_results
from fed_speaker_monitor_v2.aggregation.member import aggregate_member_stances
from fed_speaker_monitor_v2.collectors.article_text import extract_article_text
from fed_speaker_monitor_v2.collectors.fed_news import collect_fed_news_for_members
from fed_speaker_monitor_v2.collectors.finlight_news import collect_finlight_news_for_members
from fed_speaker_monitor_v2.collectors.fed_official import collect_fed_official
from fed_speaker_monitor_v2.collectors.fed_regional import collect_regional_official
from fed_speaker_monitor_v2.collectors.incremental import collect_regional_incremental
from fed_speaker_monitor_v2.collectors.reuters_stance import update_reuters_stance
from fed_speaker_monitor_v2.collectors.fed_speeches import enrich_fed_documents
from fed_speaker_monitor_v2.config import FED_MEMBERS, RESULTS_DIR
from fed_speaker_monitor_v2.llm.stance import (
    analyze_segments,
    analyze_news_events,
)
from fed_speaker_monitor_v2.llm.news_relevance import analyze_news_relevance_segments
from fed_speaker_monitor_v2.processors.dedup import deduplicate_documents
from fed_speaker_monitor_v2.processors.pre_llm_dedup import deduplicate_raw_anchor
from fed_speaker_monitor_v2.llm.relevance_embedding import validate_relevance_cluster
from fed_speaker_monitor_v2.processors.final_event_embedding import build_final_events
from fed_speaker_monitor_v2.processors.document import process_documents
from fed_speaker_monitor_v2.processors.segments import segment_documents
from fed_speaker_monitor_v2.models import Segment


TARGET_YEAR = 2026


# ============================================================
# OBVIOUS JUNK FILTER
# ============================================================

JUNK_TITLE_TERMS = (
    "view photos",
    "listen to event audio",
    "speaker request",
    "request form",
    "speakers bureau",
    "money museum",
)

JUNK_URL_TERMS = (
    "/speaker-request",
    "/request-form",
    "/speakers-bureau",
)


def _is_obvious_junk(
    document,
) -> bool:
    """
    명백한 navigation / utility document만 제거한다.

    정책 관련성은 여기서 판단하지 않는다.
    Hawk/Dove relevance는 LLM이 담당한다.
    """
    title = (
        getattr(
            document,
            "title",
            "",
        )
        or ""
    ).strip()

    url = (
        getattr(
            document,
            "url",
            "",
        )
        or ""
    ).strip().lower()

    title_lower = title.lower()

    # 제목이 없어도 본문이 충분하면 유지
    # Jeffrey Schmid처럼 title이 비어 있어도 실제 speech body가 있으면 살린다.
    if not title:
        text = (
            getattr(
                document,
                "text",
                "",
            )
            or ""
        ).strip()

        if len(text) < 300:
            return True

    if any(
        term in title_lower
        for term in JUNK_TITLE_TERMS
    ):
        return True

    if any(
        term in url
        for term in JUNK_URL_TERMS
    ):
        return True

    return False


def _filter_obvious_junk(
    documents,
):
    kept = []
    removed = []

    for document in documents:
        if _is_obvious_junk(
            document
        ):
            removed.append(
                document
            )
        else:
            kept.append(
                document
            )

    if removed:
        print(
            f"      Junk removed: "
            f"{len(removed)}"
        )

        for document in removed[:10]:
            print(
                "        -",
                getattr(
                    document,
                    "speaker",
                    "",
                ),
                "|",
                getattr(
                    document,
                    "title",
                    "",
                )[:90],
            )

        if len(removed) > 10:
            print(
                f"        ... +"
                f"{len(removed) - 10} more"
            )

    return kept


# ============================================================
# DATE
# ============================================================

def _year_from_iso(
    value,
):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        ).year

    except ValueError:
        return None


def _filter_year(
    documents,
    target_year,
):
    results = []

    for document in documents:
        year = _year_from_iso(
            getattr(
                document,
                "published_at",
                None,
            )
        )

        if (
            year is not None
            and year != target_year
        ):
            continue

        results.append(
            document
        )

    return results


# ============================================================
# GOOGLE
# ============================================================

def _enrich_google_documents(
    documents,
):
    results = []
    total = len(documents)

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"[GOOGLE BODY {index}/{total}] "
            f"{document.speaker or 'Unknown'} | "
            f"{document.title[:80]}"
        )

        try:
            document.text = (
                extract_article_text(
                    document.url
                )
            )

        except Exception as exc:
            print(
                f"    -> failed: {exc}"
            )
            document.text = ""

        document.fetch_ok = bool(
            document.text
        )

        if document.fetch_ok:
            results.append(
                document
            )

    return results


# ============================================================
# COVERAGE
# ============================================================

def _source_group(
    source,
):
    source = (
        source
        or ""
    ).lower()

    if source.startswith(
        "regional_"
    ):
        return "REGIONAL"

    if source in {
        "fed_speech",
        "fed_testimony",
    }:
        return "BOARD"

    if source == "google_news":
        return "GOOGLE"

    if source == "wire":
        return "WIRE"

    return (
        source.upper()
        or "UNKNOWN"
    )


def _build_coverage(
    documents,
):
    coverage = defaultdict(
        Counter
    )

    for document in documents:
        member = (
            document.speaker
            or "Unknown"
        )

        group = _source_group(
            document.source
        )

        coverage[
            member
        ][group] += 1

    rows = []

    for member in FED_MEMBERS:
        counts = coverage.get(
            member,
            Counter(),
        )

        rows.append({
            "member":
                member,

            "board":
                counts.get(
                    "BOARD",
                    0,
                ),

            "regional":
                counts.get(
                    "REGIONAL",
                    0,
                ),

            "google":
                counts.get(
                    "GOOGLE",
                    0,
                ),

            "wire":
                counts.get(
                    "WIRE",
                    0,
                ),

            "total":
                sum(
                    counts.values()
                ),
        })

    return rows


def _print_coverage(
    coverage_rows,
):
    print()
    print(
        "=" * 90
    )
    print(
        "SOURCE COVERAGE"
    )
    print(
        "=" * 90
    )

    for row in coverage_rows:
        print(
            f"{row['member']:<22} "
            f"BOARD={row['board']:<3} "
            f"REGIONAL={row['regional']:<3} "
            f"GOOGLE={row['google']:<3} "
            f"WIRE={row['wire']:<3} "
            f"TOTAL={row['total']}"
        )


# ============================================================
# SERIALIZE
# ============================================================

def _document_to_dict(
    document,
):
    row = asdict(
        document
    )

    row["date_type"] = getattr(
        document,
        "date_type",
        None,
    )

    return row


def _write_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )



# ============================================================
# NEWS LAYER
# ============================================================

def _news_segment_id(
    document,
) -> str:
    """
    URL 기반의 stable segment_id.

    같은 기사를 다음 실행에서 다시 수집해도
    동일 ID가 생성되어 LLM cache를 재사용한다.
    """
    speaker = re.sub(
        r"[^a-z0-9]+",
        "_",
        (
            document.speaker
            or "unknown"
        ).lower(),
    ).strip("_")

    digest = hashlib.sha1(
        document.url.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    return (
        f"news_{speaker}_{digest}"
    )


def _build_news_segments(
    documents,
):
    """
    뉴스는 speech처럼 여러 paragraph로 쪼개지 않는다.

    1 representative news event = 1 Segment

    LLM에는 title + Finlight summary를 함께 준다.
    """
    segments = []

    for document in documents:
        title = (
            getattr(
                document,
                "title",
                "",
            )
            or ""
        ).strip()

        summary = (
            getattr(
                document,
                "text",
                "",
            )
            or ""
        ).strip()

        text = (
            f"Title: {title}\\n\\n"
            f"Summary: {summary}"
        ).strip()

        if not title:
            continue

        segments.append(
            Segment(
                segment_id=(
                    _news_segment_id(
                        document
                    )
                ),
                document_url=document.url,
                text=text,
                speaker=document.speaker,
            )
        )

    return segments


def _news_member_coverage(
    documents,
):
    return {
        document.speaker
        for document in documents
        if document.speaker
    }


def _build_news_event_segments(events):
    """Final Event target_text -> News Stance Segment."""
    results = []
    for event in events:
        event_id = str(event.get("event_id", "") or "").strip()
        speaker = str(event.get("speaker", "") or "").strip()
        target_text = str(event.get("target_text", "") or "").strip()
        if event_id and speaker and target_text:
            results.append(Segment(segment_id=event_id, document_url="", text=target_text, speaker=speaker))
    return results


def _build_news_event_dates(events, documents, segments):
    """Final Event representative article date -> Macro as_of."""
    segment_to_url = {s.segment_id: s.document_url for s in segments}
    document_by_url = {d.url: d for d in documents}
    as_of_by_id = {}
    for event in events:
        event_id = event.get("event_id")
        representative_id = event.get("representative_segment_id")
        if not event_id or not representative_id:
            continue
        document = document_by_url.get(segment_to_url.get(representative_id))
        if document is not None and document.published_at:
            as_of_by_id[event_id] = document.published_at
    return as_of_by_id


def _merge_news_history(events, documents, segments, stance_results):
    """Preserve backfilled event history and merge current Final Events by event_id."""
    path = RESULTS_DIR / "news_history.json"
    existing = []
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                existing = value
        except (OSError, json.JSONDecodeError):
            existing = []

    stance_by_id = {r.segment_id: r for r in stance_results}
    segment_to_url = {s.segment_id: s.document_url for s in segments}
    document_by_url = {d.url: d for d in documents}
    current_rows = []

    for event in events:
        event_id = event.get("event_id")
        stance = stance_by_id.get(event_id)
        if not event_id or stance is None:
            continue
        document = document_by_url.get(segment_to_url.get(event.get("representative_segment_id")))
        if document is None:
            continue
        current_rows.append({
            "segment_id": event_id, "event_id": event_id, "speaker": event.get("speaker"),
            "published_at": document.published_at, "title": document.title, "summary": document.text,
            "target_text": event.get("target_text", ""), "url": document.url, "source": document.source,
            "article_count": event.get("article_count", 1), "segment_ids": event.get("segment_ids", []),
            "policy_relevant": stance.policy_relevant, "stance": stance.stance, "score": stance.score,
            "bis_stance": stance.bis_stance, "policy_action": stance.policy_action,
            "signal_strength": stance.signal_strength, "stance_driver": stance.stance_driver,
            "policy_bearing_phrase": stance.policy_bearing_phrase,
            "speaker_evidence_type": stance.speaker_evidence_type, "directness": stance.directness,
            "content_type": stance.content_type, "temporal": stance.temporal, "uncertainty": stance.uncertainty,
            "evidence_confidence": stance.evidence_confidence, "text_sufficiency": stance.text_sufficiency,
            "evidence": stance.evidence, "reasoning": stance.reasoning,
            "macro_calibrated": stance.macro_calibrated, "macro_background": stance.macro_background,
        })

    by_id = {}
    for row in existing:
        if isinstance(row, dict):
            row_id = row.get("event_id") or row.get("segment_id")
            if row_id:
                by_id[str(row_id)] = row
    for row in current_rows:
        by_id[row["event_id"]] = row
    merged = list(by_id.values())
    merged.sort(key=lambda row: str(row.get("published_at", "")))
    _write_json(path, merged)
    return merged


def _get_news_collection_lookback_days():
    """
    최초 News 실행:
    2026-01-01부터 현재까지 backfill.

    backfill 완료 후:
    news_history.json의 마지막 날짜부터 증분 수집.

    최근 기사 누락 방지를 위해 2일 overlap을 둔다.
    """

    backfill_marker = (
        RESULTS_DIR
        / "news_backfill_2026.done"
    )

    today = datetime.now().date()

    # --------------------------------------------------------
    # Backfill된 event history가 있으면 바로 incremental.
    # marker가 없어도 history의 마지막 날짜부터 2일 overlap 수집.
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR
        / "news_history.json"
    )

    if not history_path.exists():
        if not backfill_marker.exists():
            start_date = datetime(2026, 1, 1).date()
            return (today - start_date).days + 1
        return 14

    try:
        history = json.loads(
            history_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return 14

    dates = []

    for row in history:
        value = row.get(
            "published_at"
        )

        if not value:
            continue

        try:
            dt = datetime.fromisoformat(
                str(
                    value
                ).replace(
                    "Z",
                    "+00:00",
                )
            )

            dates.append(
                dt.date()
            )

        except ValueError:
            continue

    if not dates:
        return 14

    latest_date = max(
        dates
    )

    # 2일 overlap
    lookback_days = (
        today
        - latest_date
    ).days + 2

    return max(
        2,
        lookback_days,
    )

def _run_news_layer(
    *,
    lookback_days=14,
    run_llm=True,
):
    """
    News pipeline.

    PRIMARY:
        Finlight

    FALLBACK:
        Google News RSS only for members with no Finlight result

    순서:
        collect
        -> process
        -> ① Raw Anchor dedup (.78 / 4D)
        -> representative documents
        -> segment
        -> ② Relevance LLM
        -> ② Relevance embedding validation
        -> ③ speaker-level full-text Final Event
        -> ④ News Stance + conditional Macro calibration
        -> event-based history merge

    저장:
    - news_documents_before_dedup.json
    - news_documents.json
    - news_segments.json
    - news_relevance.json
    - news_events.json
    - news_segment_stance.json
    - news_history.json

    News는 Final Event 이후 전용 Stance와 event-based history까지 실행한다.
    """

    print()
    print("=" * 90)
    print("NEWS LAYER")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. FINLIGHT PRIMARY
    # --------------------------------------------------------

    collection_lookback_days = (
        _get_news_collection_lookback_days()
    )

    print(
        f"[NEWS 1/7] Collecting Finlight "
        f"(lookback={collection_lookback_days}d)..."
    )

    news_documents = (
        collect_finlight_news_for_members(
            FED_MEMBERS,
            lookback_days=collection_lookback_days,
        )
    )

    finlight_covered = (
        _news_member_coverage(
            news_documents
        )
    )

    fallback_members = [
        member
        for member in FED_MEMBERS
        if member not in finlight_covered
    ]

    print(
        f"      Finlight raw: "
        f"{len(news_documents)}"
    )

    print(
        f"      Finlight covered: "
        f"{len(finlight_covered)}/"
        f"{len(FED_MEMBERS)}"
    )

    # --------------------------------------------------------
    # 2. GOOGLE FALLBACK
    # --------------------------------------------------------

    if fallback_members:
        print(
            f"[NEWS 2/7] Google fallback "
            f"for {len(fallback_members)} "
            f"member(s)..."
        )

        google_documents = (
            collect_fed_news_for_members(
                fallback_members,
                lookback_days=collection_lookback_days,
            )
        )

        news_documents.extend(
            google_documents
        )

    else:
        print(
            "[NEWS 2/7] Google fallback: "
            "not needed"
        )

    # --------------------------------------------------------
    # RAW SNAPSHOT
    # --------------------------------------------------------

    _write_json(
        RESULTS_DIR
        / "news_documents_before_dedup.json",
        [
            _document_to_dict(document)
            for document in news_documents
        ],
    )

    # --------------------------------------------------------
    # 3. PROCESS + ① RAW ANCHOR DEDUP
    # --------------------------------------------------------

    print(
        "[NEWS 3/7] Processing + "
        "Raw Anchor dedup..."
    )

    news_documents = (
        process_documents(
            news_documents
        )
    )

    news_documents = [
        document
        for document in news_documents
        if (
            document.fetch_ok
            and document.text
            and document.speaker
        )
    ]

    before_raw_anchor = len(
        news_documents
    )

    news_documents = (
        deduplicate_raw_anchor(
            news_documents
        )
    )

    print(
        f"      Before Raw Anchor: "
        f"{before_raw_anchor}"
    )

    print(
        f"      Representatives : "
        f"{len(news_documents)}"
    )

    news_segments = (
        _build_news_segments(
            news_documents
        )
    )

    print(
        f"      News segments   : "
        f"{len(news_segments)}"
    )

    _write_json(
        RESULTS_DIR
        / "news_documents.json",
        [
            _document_to_dict(
                document
            )
            for document
            in news_documents
        ],
    )

    _write_json(
        RESULTS_DIR
        / "news_segments.json",
        [
            asdict(
                segment
            )
            for segment
            in news_segments
        ],
    )

    if not run_llm:
        print(
            "[NEWS 4/7] Relevance LLM: SKIPPED "
            "(run_llm=False)"
        )

        return {
            "documents":
                news_documents,
            "segments":
                news_segments,
            "events":
                [],
            "segment_stance":
                [],
            "members":
                [],
        }

    # --------------------------------------------------------
    # 4. ② RELEVANCE LLM
    # --------------------------------------------------------

    print(
        "[NEWS 4/7] Relevance LLM..."
    )

    relevance_rows = (
        analyze_news_relevance_segments(
            news_segments,
            cache_path=(
                RESULTS_DIR
                / "news_relevance.json"
            ),
        )
    )

    # Relevance cache 재사용은 유지하되,
    # downstream에는 이번 실행의 News segment만 전달한다.
    current_segment_ids = {
        segment.segment_id
        for segment in news_segments
    }

    relevance_rows = [
        row
        for row in relevance_rows
        if (
            isinstance(row, dict)
            and row.get("segment_id")
            in current_segment_ids
        )
    ]

    _write_json(
        RESULTS_DIR
        / "news_relevance.json",
        relevance_rows,
    )

    passed_rows = [
        row
        for row in relevance_rows
        if (
            isinstance(row, dict)
            and row.get("passed") is True
        )
    ]

    print(
        f"      Relevance PASS: "
        f"{len(passed_rows)}/"
        f"{len(relevance_rows)}"
    )

    # --------------------------------------------------------
    # 5. ② RELEVANCE EMBEDDING VALIDATION
    # --------------------------------------------------------

    print(
        "[NEWS 5/7] Relevance embedding "
        "validation..."
    )

    speaker_rows = defaultdict(
        list
    )

    for row in passed_rows:
        speaker = str(
            row.get(
                "speaker",
                "",
            )
            or ""
        ).strip()

        if not speaker:
            continue

        speaker_rows[
            speaker
        ].append(
            row
        )

    validated_rows = []

    for speaker, rows in (
        speaker_rows.items()
    ):

        validated_clusters = (
            validate_relevance_cluster(
                rows
            )
        )

        print(
            f"      {speaker}: "
            f"PASS={len(rows)} "
            f"validated_clusters="
            f"{len(validated_clusters)}"
        )

        for cluster in validated_clusters:
            for index in cluster:
                validated_rows.append(
                    rows[index]
                )

    print(
        f"      Validated rows: "
        f"{len(validated_rows)}"
    )

    # --------------------------------------------------------
    # 6. ③ FINAL EVENT FULL-TEXT CHECK
    # --------------------------------------------------------

    print(
        "[NEWS 6/7] Speaker-level "
        "Final Event full-text check..."
    )

    news_events = (
        build_final_events(
            validated_rows
        )
    )

    _write_json(
        RESULTS_DIR
        / "news_events.json",
        news_events,
    )

    print(
        f"      Final events: "
        f"{len(news_events)}"
    )


    # --------------------------------------------------------
    # 7. ④ NEWS STANCE + MACRO
    # --------------------------------------------------------

    print("[NEWS 7/7] News Stance + Macro calibration...")

    event_segments = _build_news_event_segments(news_events)
    as_of_by_id = _build_news_event_dates(news_events, news_documents, news_segments)

    news_stance_results = analyze_news_events(
        event_segments,
        cache_path=(RESULTS_DIR / "news_segment_stance.json"),
        as_of_by_id=as_of_by_id,
    )

    _write_json(
        RESULTS_DIR / "news_segment_stance.json",
        [asdict(result) for result in news_stance_results],
    )

    macro_count = sum(1 for result in news_stance_results if result.macro_calibrated)
    history = _merge_news_history(
        news_events, news_documents, news_segments, news_stance_results
    )

    print(f"      News stance      : {len(news_stance_results)}")
    print(f"      Macro calibrated : {macro_count}")
    print(f"      News history     : {len(history)}")

    print()
    print("=" * 90)
    print("NEWS ①→②→③→④ COMPLETE")
    print("=" * 90)
    print(
        f"Raw processed={before_raw_anchor} | "
        f"Representatives={len(news_documents)} | "
        f"Relevance PASS={len(passed_rows)} | "
        f"Validated={len(validated_rows)} | "
        f"Final Events={len(news_events)} | "
        f"Stance={len(news_stance_results)} | "
        f"History={len(history)}"
    )
    print("=" * 90)

    return {
        "documents": news_documents,
        "segments": news_segments,
        "events": news_events,
        "segment_stance": news_stance_results,
        "members": [],
    }


# ============================================================
# MAIN
# ============================================================

def run_pipeline(
    *,
    target_year=TARGET_YEAR,
    include_google=False,
    include_news=True,
    news_lookback_days=14,
    run_llm=True,
    incremental=True,
):
    # Reuters stance is an external benchmark only.
    # Refresh it on every pipeline run; collector keeps the previous JSON on failure.
    print()
    print("[REUTERS] Refreshing Reuters stance benchmark...")
    update_reuters_stance()

    # --------------------------------------------------------
    # 1. BOARD
    # --------------------------------------------------------

    print()
    print(
        "[1/8] Collecting Board official RSS..."
    )

    board_raw = collect_fed_official(
        lookback_days=400
    )

    print(
        f"      RSS metadata: "
        f"{len(board_raw)}"
    )

    print(
        "      Enriching Board "
        "speech/testimony bodies..."
    )

    board_documents = (
        enrich_fed_documents(
            board_raw
        )
    )

    board_documents = [
        document
        for document in board_documents
        if (
            document.fetch_ok
            and document.text
        )
    ]

    board_documents = _filter_year(
        board_documents,
        target_year,
    )

    print(
        f"      Board usable: "
        f"{len(board_documents)}"
    )

    # --------------------------------------------------------
    # 2. REGIONAL
    # --------------------------------------------------------

    print()
    print(
        f"[2/8] Collecting Regional Fed "
        f"official documents for "
        f"{target_year}..."
    )

    if incremental:
        print(
            "      Mode: INCREMENTAL "
            "(reuse cached Regional documents)"
        )
        regional_documents = (
            collect_regional_incremental(
                target_year=target_year
            )
        )
    else:
        print(
            "      Mode: FULL REFRESH"
        )
        regional_documents = (
            collect_regional_official(
                target_year=target_year
            )
        )

    print(
        f"      Regional usable: "
        f"{len(regional_documents)}"
    )

    # --------------------------------------------------------
    # 3. MERGE + JUNK + DEDUP
    # --------------------------------------------------------

    print()
    print(
        "[3/8] Merging / filtering / "
        "deduplicating..."
    )

    documents = (
        board_documents
        +
        regional_documents
    )

    print(
        f"      Merged: "
        f"{len(documents)}"
    )

    # 명백한 junk를 dedup 전에 제거
    documents = (
        _filter_obvious_junk(
            documents
        )
    )

    print(
        f"      After junk filter: "
        f"{len(documents)}"
    )

    before_dedup = len(
        documents
    )

    documents = (
        deduplicate_documents(
            documents
        )
    )

    print(
        f"      Before dedup: "
        f"{before_dedup}"
    )

    print(
        f"      After dedup : "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # 4. NORMALIZE
    # --------------------------------------------------------

    print()
    print(
        "[4/8] Normalizing documents..."
    )

    documents = (
        process_documents(
            documents
        )
    )

    documents = [
        document
        for document in documents
        if (
            document.fetch_ok
            and document.text
            and document.speaker
        )
    ]

    # normalize 이후 title이 사라진 문서도 다시 얇게 확인
    documents = (
        _filter_obvious_junk(
            documents
        )
    )

    print(
        f"      Normalized usable: "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # COVERAGE BEFORE GOOGLE
    # --------------------------------------------------------

    coverage_before_google = (
        _build_coverage(
            documents
        )
    )

    _print_coverage(
        coverage_before_google
    )

    uncovered_members = [
        row["member"]
        for row in coverage_before_google
        if row["total"] == 0
    ]

    print()
    print(
        "Uncovered members:",
        len(
            uncovered_members
        ),
    )

    for member in uncovered_members:
        print(
            "  -",
            member,
        )

    # --------------------------------------------------------
    # 5. GOOGLE OPTIONAL
    # --------------------------------------------------------

    print()

    if (
        include_google
        and uncovered_members
    ):
        print(
            f"[5/8] Google fallback for "
            f"{len(uncovered_members)} "
            f"uncovered member(s)..."
        )

        try:
            google_raw = (
                collect_fed_news_for_members(
                    uncovered_members,
                    lookback_days=400,
                )
            )
        except TypeError:
            google_raw = (
                collect_fed_news_for_members(
                    uncovered_members
                )
            )

        google_raw = _filter_year(
            google_raw,
            target_year,
        )

        google_documents = (
            _enrich_google_documents(
                google_raw
            )
        )

        google_documents = (
            process_documents(
                google_documents
            )
        )

        google_documents = (
            _filter_obvious_junk(
                google_documents
            )
        )

        documents.extend(
            google_documents
        )

        documents = (
            deduplicate_documents(
                documents
            )
        )

        print(
            f"      Google usable: "
            f"{len(google_documents)}"
        )

    else:
        print(
            "[5/8] Google fallback: OFF"
        )

    # --------------------------------------------------------
    # FINAL COVERAGE
    # --------------------------------------------------------

    coverage = (
        _build_coverage(
            documents
        )
    )

    _print_coverage(
        coverage
    )

    # --------------------------------------------------------
    # 6. SEGMENTS
    # --------------------------------------------------------

    print()
    print(
        "[6/8] Building segments..."
    )

    segments = (
        segment_documents(
            documents
        )
    )

    print(
        f"      Documents: "
        f"{len(documents)}"
    )

    print(
        f"      Segments : "
        f"{len(segments)}"
    )

    # --------------------------------------------------------
    # SAVE BEFORE LLM
    # --------------------------------------------------------

    _write_json(
        RESULTS_DIR
        / "documents.json",
        [
            _document_to_dict(
                document
            )
            for document
            in documents
        ],
    )

    _write_json(
        RESULTS_DIR
        / "segments.json",
        [
            asdict(
                segment
            )
            for segment
            in segments
        ],
    )

    _write_json(
        RESULTS_DIR
        / "coverage.json",
        coverage,
    )

    news_result = {
        "documents": [],
        "segments": [],
        "segment_stance": [],
        "members": [],
    }

    if include_news:
        news_result = _run_news_layer(
            lookback_days=news_lookback_days,
            run_llm=run_llm,
        )

    if not run_llm:
        print()
        print(
            "[STOP] run_llm=False"
        )

        print(
            f"FINAL CHECK | "
            f"documents={len(documents)} | "
            f"segments={len(segments)}"
        )

        return {
            "documents":
                documents,

            "segments":
                segments,

            "coverage":
                coverage,

            "segment_stance":
                [],

            "members":
                [],

            "final":
                [],

            "news":
                news_result,
        }

    # --------------------------------------------------------
    # 7. LLM
    # --------------------------------------------------------

    print()
    print(
        "[7/8] Running LLM stance analysis..."
    )

    stance_results = (
        analyze_segments(
            segments,
            cache_path=(
                RESULTS_DIR
                / "segment_stance.json"
            ),
        )
    )

    print(
        f"      LLM results: "
        f"{len(stance_results)}"
    )

    # --------------------------------------------------------
    # 8. AGGREGATION
    # --------------------------------------------------------

    print()
    print(
        "[8/8] Aggregating member results..."
    )

    member_results = (
        aggregate_member_stances(
            documents,
            segments,
            stance_results,
        )
    )

    final_results = (
        build_final_results(
            documents,
            member_results,
        )
    )

    _write_json(
        RESULTS_DIR
        / "segment_stance.json",
        [
            asdict(
                result
            )
            for result
            in stance_results
        ],
    )

    _write_json(
        RESULTS_DIR
        / "member_stance.json",
        [
            asdict(
                result
            )
            for result
            in member_results
        ],
    )

    _write_json(
        RESULTS_DIR
        / "final_results.json",
        [
            asdict(
                result
            )
            for result
            in final_results
        ],
    )

    print()
    print(
        "=" * 90
    )
    print(
        "DONE"
    )
    print(
        f"documents={len(documents)} | "
        f"segments={len(segments)} | "
        f"members={len(member_results)}"
    )
    print(
        "=" * 90
    )

    return {
        "documents":
            documents,

        "segments":
            segments,

        "coverage":
            coverage,

        "segment_stance":
            stance_results,

        "members":
            member_results,

        "final":
            final_results,

        "news":
            news_result,
    }


if __name__ == "__main__":
    run_pipeline(
        target_year=TARGET_YEAR,
        include_google=False,
        include_news=True,
        news_lookback_days=14,
        run_llm=True,
        incremental=True,
    )