from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from fed_speaker_monitor_v2.config import RESULTS_DIR
from fed_speaker_monitor_v2.models import (
    Document,
    Segment,
)

from fed_speaker_monitor_v2.pipeline import (
    _news_segment_id,
)

from fed_speaker_monitor_v2.processors.pre_llm_dedup import (
    deduplicate_raw_anchor,
)

from fed_speaker_monitor_v2.llm.news_relevance import (
    analyze_news_relevance_segments,
)

from fed_speaker_monitor_v2.llm.relevance_embedding import (
    validate_relevance_cluster,
)

from fed_speaker_monitor_v2.processors.final_event_embedding import (
    build_final_events,
)

from fed_speaker_monitor_v2.llm.stance import (
    analyze_news_events,
)


# ============================================================
# CONFIG
# ============================================================

# 처음에는 반드시 True.
TEST_MODE = False

# TEST_MODE=True일 때 legacy 앞 N개만 사용.
TEST_LIMIT = 100

LEGACY_HISTORY_PATH = (
    RESULTS_DIR
    / "news_history_legacy_before_event_dedup.json"
)

RELEVANCE_CACHE_PATH = (
    RESULTS_DIR
    / "news_relevance.json"
)

TEST_OUTPUT_PATH = (
    RESULTS_DIR
    / "news_history_backfill_test.json"
)

TEST_STANCE_CACHE_PATH = (
    RESULTS_DIR
    / "news_backfill_stance_test.json"
)

FINAL_OUTPUT_PATH = (
    RESULTS_DIR
    / "news_history.json"
)

FINAL_STANCE_CACHE_PATH = (
    RESULTS_DIR
    / "news_segment_stance.json"
)


# ============================================================
# JSON
# ============================================================

def load_json(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        list,
    ):
        raise TypeError(
            f"Expected list: {path}"
        )

    return [
        row
        for row in value
        if isinstance(
            row,
            dict,
        )
    ]


def write_json(
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
# LEGACY HISTORY -> DOCUMENT
# ============================================================

def build_documents(
    rows: list[dict],
) -> list[Document]:
    """
    구 news_history row를 새 News pipeline이 사용하는
    Document 형태로 복원한다.

    원본 legacy 파일은 수정하지 않는다.
    """

    documents = []

    for row in rows:

        url = str(
            row.get(
                "url",
                "",
            )
            or ""
        ).strip()

        speaker = str(
            row.get(
                "speaker",
                "",
            )
            or ""
        ).strip()

        title = str(
            row.get(
                "title",
                "",
            )
            or ""
        ).strip()

        summary = str(
            row.get(
                "summary",
                "",
            )
            or ""
        ).strip()

        published_at = row.get(
            "published_at"
        )

        source = str(
            row.get(
                "source",
                "legacy_news",
            )
            or "legacy_news"
        ).strip()

        if not url:
            continue

        if not speaker:
            continue

        if not title:
            continue

        if not summary:
            continue

        documents.append(
            Document(
                source=source,
                title=title,
                url=url,
                published_at=published_at,
                speaker=speaker,
                text=summary,
                fetch_ok=True,
            )
        )

    return documents


# ============================================================
# DOCUMENT -> SEGMENT
# ============================================================

def build_segments(
    documents: list[Document],
) -> list[Segment]:

    segments = []

    for document in documents:

        text = (
            f"Title: {document.title}\n\n"
            f"Summary: {document.text}"
        ).strip()

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


# ============================================================
# RELEVANCE EMBEDDING
# ============================================================

def validate_relevance(
    relevance_rows: list[dict],
) -> list[dict]:

    passed_rows = [
        row
        for row in relevance_rows
        if (
            isinstance(
                row,
                dict,
            )
            and row.get(
                "passed"
            ) is True
        )
    ]

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

        clusters = (
            validate_relevance_cluster(
                rows
            )
        )

        print(
            f"[VALIDATE] "
            f"{speaker} | "
            f"PASS={len(rows)} | "
            f"clusters={len(clusters)}"
        )

        for cluster in clusters:

            for index in cluster:

                validated_rows.append(
                    rows[index]
                )

    return validated_rows


# ============================================================
# EVENT -> STANCE SEGMENT
# ============================================================

def build_event_segments(
    events: list[dict],
) -> list[Segment]:
    """
    Final Event의 대표 target_text를
    News Stance 입력으로 변환한다.
    """

    segments = []

    for event in events:

        event_id = str(
            event.get(
                "event_id",
                "",
            )
            or ""
        ).strip()

        speaker = str(
            event.get(
                "speaker",
                "",
            )
            or ""
        ).strip()

        target_text = str(
            event.get(
                "target_text",
                "",
            )
            or ""
        ).strip()

        if not event_id:
            continue

        if not speaker:
            continue

        if not target_text:
            continue

        segments.append(
            Segment(
                segment_id=event_id,
                document_url="",
                text=target_text,
                speaker=speaker,
            )
        )

    return segments


# ============================================================
# EVENT DATE
# ============================================================

def build_event_dates(
    events: list[dict],
    documents: list[Document],
    segments: list[Segment],
) -> dict[str, str]:
    """
    Final Event의 representative_segment_id를 이용해
    원 기사의 published_at을 찾는다.

    News Macro Calibration의 as_of로 사용한다.
    """

    segment_to_url = {
        segment.segment_id:
            segment.document_url
        for segment in segments
    }

    document_by_url = {
        document.url:
            document
        for document in documents
    }

    as_of_by_id = {}

    for event in events:

        event_id = event.get(
            "event_id"
        )

        representative_id = (
            event.get(
                "representative_segment_id"
            )
        )

        if (
            not event_id
            or not representative_id
        ):
            continue

        url = segment_to_url.get(
            representative_id
        )

        document = (
            document_by_url.get(
                url
            )
        )

        if (
            document is None
            or not document.published_at
        ):
            continue

        as_of_by_id[
            event_id
        ] = document.published_at

    return as_of_by_id


# ============================================================
# FINAL HISTORY
# ============================================================

def build_history(
    events: list[dict],
    stance_results,
    documents: list[Document],
    segments: list[Segment],
) -> list[dict]:

    stance_by_id = {
        result.segment_id:
            result
        for result in stance_results
    }

    segment_to_url = {
        segment.segment_id:
            segment.document_url
        for segment in segments
    }

    document_by_url = {
        document.url:
            document
        for document in documents
    }

    history = []

    for event in events:

        event_id = event.get(
            "event_id"
        )

        if not event_id:
            continue

        stance = stance_by_id.get(
            event_id
        )

        if stance is None:
            continue

        representative_id = (
            event.get(
                "representative_segment_id"
            )
        )

        url = segment_to_url.get(
            representative_id
        )

        document = (
            document_by_url.get(
                url
            )
        )

        if document is None:
            continue

        history.append(
            {
                "segment_id":
                    event_id,

                "event_id":
                    event_id,

                "speaker":
                    event.get(
                        "speaker"
                    ),

                "published_at":
                    document.published_at,

                "title":
                    document.title,

                "summary":
                    document.text,

                "target_text":
                    event.get(
                        "target_text",
                        ""
                    ),

                "url":
                    document.url,

                "source":
                    document.source,

                "article_count":
                    event.get(
                        "article_count",
                        1,
                    ),

                "segment_ids":
                    event.get(
                        "segment_ids",
                        [],
                    ),

                # -------------------------
                # Stance
                # -------------------------

                "policy_relevant":
                    stance.policy_relevant,

                "stance":
                    stance.stance,

                "score":
                    stance.score,

                "bis_stance":
                    stance.bis_stance,

                "policy_action":
                    stance.policy_action,

                "signal_strength":
                    stance.signal_strength,

                "stance_driver":
                    stance.stance_driver,

                "policy_bearing_phrase":
                    stance.policy_bearing_phrase,

                "speaker_evidence_type":
                    stance.speaker_evidence_type,

                "directness":
                    stance.directness,

                "content_type":
                    stance.content_type,

                "temporal":
                    stance.temporal,

                "uncertainty":
                    stance.uncertainty,

                "evidence_confidence":
                    stance.evidence_confidence,

                "text_sufficiency":
                    stance.text_sufficiency,

                "evidence":
                    stance.evidence,

                "reasoning":
                    stance.reasoning,

                # -------------------------
                # Macro calibration
                # -------------------------

                "macro_calibrated":
                    stance.macro_calibrated,

                "macro_background":
                    stance.macro_background,
            }
        )

    history.sort(
        key=lambda row:
            str(
                row.get(
                    "published_at",
                    "",
                )
            )
    )

    return history


# ============================================================
# MAIN
# ============================================================

def run_backfill():

    print()
    print("=" * 100)
    print("NEWS HISTORY BACKFILL")
    print("=" * 100)

    print(
        "MODE:",
        (
            "TEST"
            if TEST_MODE
            else "FULL"
        ),
    )

    # --------------------------------------------------------
    # 0. LOAD LEGACY
    # --------------------------------------------------------

    legacy_rows = load_json(
        LEGACY_HISTORY_PATH
    )

    print(
        "LEGACY ROWS:",
        len(legacy_rows)
    )

    if TEST_MODE:

        legacy_rows = (
            legacy_rows[
                :TEST_LIMIT
            ]
        )

        print(
            "TEST LIMIT:",
            len(legacy_rows)
        )

    # --------------------------------------------------------
    # 1. DOCUMENT
    # --------------------------------------------------------

    documents = build_documents(
        legacy_rows
    )

    print()
    print(
        "[1] DOCUMENTS:",
        len(documents)
    )

    # --------------------------------------------------------
    # 2. RAW ANCHOR
    # --------------------------------------------------------

    before_anchor = len(
        documents
    )

    documents = (
        deduplicate_raw_anchor(
            documents
        )
    )

    print(
        "[2] RAW ANCHOR:",
        before_anchor,
        "->",
        len(documents),
    )

    # --------------------------------------------------------
    # 3. SEGMENTS
    # --------------------------------------------------------

    segments = build_segments(
        documents
    )

    print(
        "[3] SEGMENTS:",
        len(segments)
    )

    # --------------------------------------------------------
    # 4. RELEVANCE
    # --------------------------------------------------------

    print()
    print(
        "[4] RELEVANCE"
    )

    relevance_rows = (
        analyze_news_relevance_segments(
            segments,
            cache_path=(
                RELEVANCE_CACHE_PATH
            ),
        )
    )

    # --------------------------------------------------------
    # 현재 backfill 입력 segment만 유지
    #
    # analyze_news_relevance_segments()는 cache 전체 row를
    # 반환할 수 있으므로, 기존 최신 뉴스 cache가 이번
    # TEST/FULL backfill 표본에 섞이지 않게 한다.
    # --------------------------------------------------------

    current_segment_ids = {
        segment.segment_id
        for segment in segments
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

    passed_rows = [
        row
        for row in relevance_rows
        if (
            isinstance(
                row,
                dict,
            )
            and row.get(
                "passed"
            ) is True
        )
    ]

    print(
        "    TOTAL:",
        len(relevance_rows)
    )

    print(
        "    PASS :",
        len(passed_rows)
    )

    # --------------------------------------------------------
    # 5. RELEVANCE EMBEDDING
    # --------------------------------------------------------

    print()
    print(
        "[5] RELEVANCE EMBEDDING"
    )

    validated_rows = (
        validate_relevance(
            relevance_rows
        )
    )

    print(
        "    VALIDATED:",
        len(validated_rows)
    )

    # --------------------------------------------------------
    # 6. FINAL EVENT
    # --------------------------------------------------------

    print()
    print(
        "[6] FINAL EVENT"
    )

    events = build_final_events(
        validated_rows
    )

    print(
        "    EVENTS:",
        len(events)
    )

    # --------------------------------------------------------
    # 7. EVENT STANCE INPUT
    # --------------------------------------------------------

    event_segments = (
        build_event_segments(
            events
        )
    )

    as_of_by_id = (
        build_event_dates(
            events,
            documents,
            segments,
        )
    )

    print()
    print(
        "[7] STANCE INPUT:",
        len(event_segments)
    )

    print(
        "    EVENT DATES:",
        len(as_of_by_id)
    )

    # --------------------------------------------------------
    # 8. NEWS STANCE + MACRO
    # --------------------------------------------------------

    print()
    print(
        "[8] NEWS STANCE"
    )

    stance_cache_path = (
        TEST_STANCE_CACHE_PATH
        if TEST_MODE
        else FINAL_STANCE_CACHE_PATH
    )

    stance_results = (
        analyze_news_events(
            event_segments,
            cache_path=(
                stance_cache_path
            ),
            as_of_by_id=(
                as_of_by_id
            ),
        )
    )

    macro_count = sum(
        1
        for result
        in stance_results
        if result.macro_calibrated
    )

    print(
        "    STANCE RESULTS:",
        len(stance_results)
    )

    print(
        "    MACRO CALIBRATED:",
        macro_count
    )

    # --------------------------------------------------------
    # 9. HISTORY
    # --------------------------------------------------------

    history = build_history(
        events,
        stance_results,
        documents,
        segments,
    )

    output_path = (
        TEST_OUTPUT_PATH
        if TEST_MODE
        else FINAL_OUTPUT_PATH
    )

    write_json(
        output_path,
        history,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("BACKFILL SUMMARY")
    print("=" * 100)

    print(
        "LEGACY INPUT      :",
        len(legacy_rows)
    )

    print(
        "RAW REPRESENTATIVE:",
        len(documents)
    )

    print(
        "RELEVANCE PASS    :",
        len(passed_rows)
    )

    print(
        "VALIDATED         :",
        len(validated_rows)
    )

    print(
        "FINAL EVENTS      :",
        len(events)
    )

    print(
        "STANCE RESULTS    :",
        len(stance_results)
    )

    print(
        "MACRO CALIBRATED  :",
        macro_count
    )

    print(
        "HISTORY ROWS      :",
        len(history)
    )

    print(
        "OUTPUT            :",
        output_path
    )

    print("=" * 100)

    if TEST_MODE:
        print(
            "TEST MODE ONLY."
        )
        print(
            "news_history.json was NOT overwritten."
        )

    else:
        print(
            "FULL BACKFILL COMPLETE."
        )

    print("=" * 100)


if __name__ == "__main__":
    run_backfill()