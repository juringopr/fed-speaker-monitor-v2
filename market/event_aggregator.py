import json
import re
from pathlib import Path

from fed_speaker_monitor_v2.market.event_matcher_llm import (
    match_policy_events,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

SIGNALS_PATH = (
    MARKET_DIR
    / "strong_signals.json"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "aggregated_events.json"
)


# ============================================================
# JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# BASIC
# ============================================================

def clean_text(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def safe_float(value):

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# SCORE
# ============================================================

def get_signal_score(row):
    """
    Official:
        raw score

    News:
        evidence-quality weighted score
    """

    if not row:
        return None

    source_type = row.get(
        "source_type"
    )

    if source_type == "OFFICIAL":

        value = row.get(
            "raw_score"
        )

        if value is None:
            value = row.get(
                "score"
            )

    else:

        value = row.get(
            "weighted_score"
        )

        if value is None:
            value = row.get(
                "score"
            )

    return safe_float(value)


# ============================================================
# DISPLAY TEXT
# ============================================================

def get_key_evidence(row):

    if not row:
        return None

    for field in [
        "policy_bearing_phrase",
        "evidence",
        "title",
    ]:

        value = clean_text(
            row.get(field)
        )

        if value:
            return value

    return None


def get_event_summary(row):

    if not row:
        return None

    phrase = clean_text(
        row.get(
            "policy_bearing_phrase"
        )
    )

    if phrase:
        return phrase

    evidence = clean_text(
        row.get(
            "evidence"
        )
    )

    if evidence:
        return evidence

    return clean_text(
        row.get("title")
    )


# ============================================================
# BASIC EVENT CANDIDATE
# ============================================================

def same_basic_event(
    row_a,
    row_b,
):
    """
    LLM에 보낼 후보를 제한한다.

    같은:
    - speaker
    - date
    - stance

    인 경우에만 실제 동일 event인지 LLM 판단.
    """

    if (
        row_a.get("speaker")
        != row_b.get("speaker")
    ):
        return False

    if (
        row_a.get("date")
        != row_b.get("date")
    ):
        return False

    if (
        row_a.get("stance")
        != row_b.get("stance")
    ):
        return False

    return True


# ============================================================
# LLM EVENT MATCH
# ============================================================

def is_same_event(
    row_a,
    row_b,
):
    """
    실제 동일한 Fed policy event인지
    event_matcher_llm.py로 판단한다.
    """

    if not same_basic_event(
        row_a,
        row_b,
    ):
        return False, None

    result = match_policy_events(
        row_a,
        row_b,
        use_cache=True,
    )

    return (
        result.get("result")
        == "SAME_EVENT",
        result,
    )


# ============================================================
# NEWS EVENT WRAPPER
# ============================================================

def cluster_news_signals(
    news_rows,
):
    """
    News는 upstream ①→②→③에서 이미 Final Event 단위로 확정된다.

    따라서 여기서는 News ↔ News SAME_EVENT 재판정을 하지 않는다.
    strong_signals.json의 News row 1개를 그대로 Event 1개로 유지한다.

    Official ↔ News matching은 downstream의
    find_matching_news_cluster()에서 계속 수행한다.
    """

    clusters = [
        {
            "representative": news,
            "rows": [news],
            "match_results": [],
        }
        for news in news_rows
    ]

    stats = {
        "llm_checks": 0,
        "cache_checks": 0,
        "same_matches": 0,
    }

    return (
        clusters,
        stats,
    )


# ============================================================
# NEWS CLUSTER SCORE
# ============================================================

def build_news_cluster(
    cluster,
):
    """
    동일 실제 발언을 보도한 News들의
    weighted score 평균.
    """

    rows = cluster[
        "rows"
    ]

    scores = []


    for row in rows:

        score = get_signal_score(
            row
        )

        if score is not None:

            scores.append(
                score
            )


    if scores:

        news_score = (
            sum(scores)
            / len(scores)
        )

    else:

        news_score = None


    # --------------------------------------------------------
    # 대표 News
    #
    # 가장 절대 score가 큰 기사 사용.
    # Event score 자체는 모든 기사 평균.
    # --------------------------------------------------------

    representative = max(
        rows,
        key=lambda row:
            abs(
                get_signal_score(row)
                or 0
            ),
    )


    return {
        "representative":
            representative,

        "rows":
            rows,

        "news_score":
            news_score,

        "news_count":
            len(rows),

        "news_scores":
            scores,

        "match_results":
            cluster.get(
                "match_results",
                [],
            ),
    }


# ============================================================
# OFFICIAL ↔ NEWS EVENT
# ============================================================

def find_matching_news_cluster(
    official,
    news_clusters,
    used_clusters,
):
    """
    Official과 News Event를 LLM으로 비교한다.

    같은 speaker/date/stance 후보만 비교.

    SAME_EVENT가 나오면 해당 News cluster를
    Official Event에 연결한다.
    """

    checks = {
        "llm":
            0,

        "cache":
            0,
    }


    for index, cluster in enumerate(
        news_clusters
    ):

        if index in used_clusters:
            continue


        representative = (
            cluster[
                "representative"
            ]
        )


        if not same_basic_event(
            official,
            representative,
        ):
            continue


        same_event, result = (
            is_same_event(
                official,
                representative,
            )
        )


        if result:

            source = result.get(
                "source"
            )

            if source == "LLM":
                checks["llm"] += 1

            elif source == "CACHE":
                checks["cache"] += 1


        if same_event:

            return (
                index,
                result,
                checks,
            )


    return (
        None,
        None,
        checks,
    )


# ============================================================
# EVENT ID
# ============================================================

def make_event_id(
    speaker,
    event_date,
    stance,
    index,
):

    speaker_text = (
        speaker
        or "unknown"
    )

    speaker_text = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        speaker_text.lower(),
    ).strip("_")


    stance_text = (
        stance
        or "unknown"
    ).lower()


    date_text = (
        event_date
        or "unknown"
    )


    return (
        f"{date_text}_"
        f"{speaker_text}_"
        f"{stance_text}_"
        f"{index:03d}"
    )


# ============================================================
# NEWS SOURCE INFO
# ============================================================

def build_news_sources(rows):

    sources = []


    for row in rows:

        sources.append(
            {
                "title":
                    row.get(
                        "title"
                    ),

                "source":
                    row.get(
                        "source"
                    ),

                "url":
                    row.get(
                        "url"
                    ),

                "segment_id":
                    row.get(
                        "segment_id"
                    ),

                "raw_score":
                    safe_float(
                        row.get(
                            "raw_score"
                        )
                    ),

                "weighted_score":
                    get_signal_score(
                        row
                    ),

                "quality_weight":
                    row.get(
                        "quality_weight"
                    ),

                "evidence":
                    row.get(
                        "evidence"
                    ),

                "policy_bearing_phrase":
                    row.get(
                        "policy_bearing_phrase"
                    ),
            }
        )


    return sources


# ============================================================
# BUILD EVENT
# ============================================================

def build_event(
    official=None,
    news_cluster=None,
    match_result=None,
    index=1,
):

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    if news_cluster:

        news_row = (
            news_cluster[
                "representative"
            ]
        )

        news_rows = (
            news_cluster[
                "rows"
            ]
        )

        news_score = (
            news_cluster[
                "news_score"
            ]
        )

        news_count = (
            news_cluster[
                "news_count"
            ]
        )

    else:

        news_row = None
        news_rows = []
        news_score = None
        news_count = 0


    # --------------------------------------------------------
    # OFFICIAL
    # --------------------------------------------------------

    official_score = (
        get_signal_score(
            official
        )
        if official
        else None
    )


    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    if (
        official_score is not None
        and news_score is not None
    ):

        match_type = (
            "OFFICIAL_NEWS_MATCHED"
        )

        combined_score = (
            official_score
            + news_score
        ) / 2


    elif official_score is not None:

        match_type = (
            "OFFICIAL_ONLY"
        )

        combined_score = (
            official_score
        )


    elif news_score is not None:

        match_type = (
            "NEWS_ONLY"
        )

        combined_score = (
            news_score
        )


    else:

        return None


    # --------------------------------------------------------
    # BASE METADATA
    # --------------------------------------------------------

    base_row = (
        official
        if official
        else news_row
    )


    speaker = base_row.get(
        "speaker"
    )

    event_date = base_row.get(
        "date"
    )

    stance = base_row.get(
        "stance"
    )


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    official_evidence = (
        get_key_evidence(
            official
        )
        if official
        else None
    )

    news_evidence = (
        get_key_evidence(
            news_row
        )
        if news_row
        else None
    )


    key_evidence = (
        official_evidence
        or news_evidence
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    event_summary = (
        get_event_summary(
            official
        )
        if official
        else None
    )


    if not event_summary:

        event_summary = (
            get_event_summary(
                news_row
            )
            if news_row
            else None
        )


    # --------------------------------------------------------
    # LLM MATCH INFO
    # --------------------------------------------------------

    llm_result = None
    llm_confidence = None
    llm_reason = None
    llm_source = None


    if match_result:

        llm_result = (
            match_result.get(
                "result"
            )
        )

        llm_confidence = (
            match_result.get(
                "confidence"
            )
        )

        llm_reason = (
            match_result.get(
                "reason"
            )
        )

        llm_source = (
            match_result.get(
                "source"
            )
        )


    # --------------------------------------------------------
    # EVENT ID
    #
    # News가 포함된 Event는 upstream Final Event의 stable
    # event_id를 그대로 보존한다.
    #
    # Official Only만 기존 방식으로 ID를 생성한다.
    # --------------------------------------------------------

    news_event_id = (
        (
            news_row.get("event_id")
            or news_row.get("segment_id")
        )
        if news_row
        else None
    )

    event_id = (
        news_event_id
        if news_event_id
        else make_event_id(
            speaker,
            event_date,
            stance,
            index,
        )
    )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    event = {

        "event_id":
            event_id,

        "date":
            event_date,

        "speaker":
            speaker,

        "stance":
            stance,

        "match_type":
            match_type,


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        "official_score":
            official_score,

        "news_score":
            news_score,

        "combined_score":
            combined_score,


        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        "official_count":
            1 if official else 0,

        "news_count":
            news_count,


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        "event_summary":
            event_summary,

        "key_evidence":
            key_evidence,

        "official_evidence":
            official_evidence,

        "news_evidence":
            news_evidence,


        # ----------------------------------------------------
        # LLM MATCH DIAGNOSTIC
        # ----------------------------------------------------

        "event_match_result":
            llm_result,

        "event_match_confidence":
            llm_confidence,

        "event_match_reason":
            llm_reason,

        "event_match_source":
            llm_source,


        # ----------------------------------------------------
        # OFFICIAL SOURCE
        # ----------------------------------------------------

        "official_title":
            (
                official.get(
                    "title"
                )
                if official
                else None
            ),

        "official_url":
            (
                official.get(
                    "url"
                )
                if official
                else None
            ),

        "official_segment_id":
            (
                official.get(
                    "segment_id"
                )
                if official
                else None
            ),


        # ----------------------------------------------------
        # NEWS SOURCES
        # ----------------------------------------------------

        "news_sources":
            build_news_sources(
                news_rows
            ),
    }


    # --------------------------------------------------------
    # REPRESENTATIVE NEWS METADATA
    # --------------------------------------------------------

    if news_row:

        event[
            "speaker_evidence_type"
        ] = news_row.get(
            "speaker_evidence_type"
        )

        event[
            "directness"
        ] = news_row.get(
            "directness"
        )

        event[
            "signal_strength"
        ] = news_row.get(
            "signal_strength"
        )

        event[
            "evidence_confidence"
        ] = news_row.get(
            "evidence_confidence"
        )

        event[
            "policy_action"
        ] = news_row.get(
            "policy_action"
        )

        event[
            "stance_driver"
        ] = news_row.get(
            "stance_driver"
        )


    return event


# ============================================================
# BUILD AGGREGATED EVENTS
# ============================================================

def build_aggregated_events():

    signals = load_json(
        SIGNALS_PATH
    )


    # ========================================================
    # 1. OFFICIAL / NEWS
    # ========================================================

    official_rows = [
        row
        for row in signals
        if row.get(
            "source_type"
        ) == "OFFICIAL"
    ]


    news_rows = [
        row
        for row in signals
        if row.get(
            "source_type"
        ) == "NEWS"
    ]


    # ========================================================
    # 2. NEWS → ACTUAL EVENT CLUSTERING
    # ========================================================

    (
        raw_clusters,
        news_match_stats,
    ) = cluster_news_signals(
        news_rows
    )


    news_clusters = [
        build_news_cluster(
            cluster
        )
        for cluster
        in raw_clusters
    ]


    # ========================================================
    # 3. OFFICIAL → NEWS EVENT MATCHING
    # ========================================================

    events = []

    used_clusters = set()

    event_index = 1

    official_llm_checks = 0
    official_cache_checks = 0


    for official in official_rows:

        (
            cluster_index,
            match_result,
            checks,
        ) = find_matching_news_cluster(
            official,
            news_clusters,
            used_clusters,
        )


        official_llm_checks += (
            checks["llm"]
        )

        official_cache_checks += (
            checks["cache"]
        )


        # ----------------------------------------------------
        # OFFICIAL + NEWS
        # ----------------------------------------------------

        if cluster_index is not None:

            used_clusters.add(
                cluster_index
            )


            event = build_event(
                official=official,
                news_cluster=(
                    news_clusters[
                        cluster_index
                    ]
                ),
                match_result=(
                    match_result
                ),
                index=event_index,
            )


        # ----------------------------------------------------
        # OFFICIAL ONLY
        # ----------------------------------------------------

        else:

            event = build_event(
                official=official,
                news_cluster=None,
                match_result=None,
                index=event_index,
            )


        if event:

            events.append(
                event
            )

            event_index += 1


    # ========================================================
    # 4. REMAINING NEWS EVENTS
    # ========================================================

    for cluster_index, cluster in enumerate(
        news_clusters
    ):

        if cluster_index in used_clusters:
            continue


        event = build_event(
            official=None,
            news_cluster=cluster,
            match_result=None,
            index=event_index,
        )


        if event:

            events.append(
                event
            )

            event_index += 1


    # ========================================================
    # 5. SORT
    # ========================================================

    events.sort(
        key=lambda row: (
            row.get(
                "date"
            ) or "",
            abs(
                safe_float(
                    row.get(
                        "combined_score"
                    )
                )
                or 0
            ),
        ),
        reverse=True,
    )


    # ========================================================
    # 6. SAVE
    # ========================================================

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            events,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # 7. SUMMARY
    # ========================================================

    matched = sum(
        row.get(
            "match_type"
        )
        == "OFFICIAL_NEWS_MATCHED"
        for row in events
    )


    official_only = sum(
        row.get(
            "match_type"
        )
        == "OFFICIAL_ONLY"
        for row in events
    )


    news_only = sum(
        row.get(
            "match_type"
        )
        == "NEWS_ONLY"
        for row in events
    )


    duplicate_news = (
        len(news_rows)
        - len(news_clusters)
    )


    total_llm_checks = (
        news_match_stats[
            "llm_checks"
        ]
        + official_llm_checks
    )


    total_cache_checks = (
        news_match_stats[
            "cache_checks"
        ]
        + official_cache_checks
    )


    print("=" * 80)
    print(
        "BUILD LLM POLICY EVENTS"
    )
    print("=" * 80)

    print(
        f"Input Signals       : "
        f"{len(signals)}"
    )

    print(
        f"Official Signals    : "
        f"{len(official_rows)}"
    )

    print(
        f"News Signals        : "
        f"{len(news_rows)}"
    )

    print()

    print(
        f"News Event Clusters : "
        f"{len(news_clusters)}"
    )

    print(
        f"Duplicate News      : "
        f"{duplicate_news}"
    )

    print()

    print(
        f"LLM Checks          : "
        f"{total_llm_checks}"
    )

    print(
        f"Cache Checks        : "
        f"{total_cache_checks}"
    )

    print()

    print(
        f"Final Events        : "
        f"{len(events)}"
    )

    print(
        f"Official + News     : "
        f"{matched}"
    )

    print(
        f"Official Only       : "
        f"{official_only}"
    )

    print(
        f"News Only           : "
        f"{news_only}"
    )

    print()

    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)


    return events


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    build_aggregated_events()