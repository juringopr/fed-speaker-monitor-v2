import json
import re
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
)

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "strong_signals.json"
)


# ============================================================
# CONFIG
# ============================================================

STRONG_THRESHOLD = 0.70


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
# OFFICIAL DATE FALLBACK
# ============================================================

def extract_official_date(url):
    """
    URL에 YYYYMMDD가 있는 경우 fallback으로 사용.

    Example:
    cook20260805a.htm
    -> 2026-08-05
    """

    match = re.search(
        r"(\d{4})(\d{2})(\d{2})",
        str(url),
    )

    if not match:
        return None

    year, month, day = match.groups()

    return f"{year}-{month}-{day}"


# ============================================================
# NEWS EVIDENCE QUALITY WEIGHT
# ============================================================

def get_quality_weight(row):
    """
    Speaker evidence 품질에 따라
    News signal 가중치를 부여한다.

    HIGH   = 1.00
    MEDIUM = 0.80
    LOW    = 0.50
    REJECT = 0.00
    """

    evidence_type = row.get(
        "speaker_evidence_type"
    )

    directness = row.get(
        "directness"
    )

    confidence = row.get(
        "evidence_confidence"
    )

    policy_phrase = row.get(
        "policy_bearing_phrase"
    )


    # --------------------------------------------------------
    # 실제 speaker evidence가 아님
    # --------------------------------------------------------

    if evidence_type in {
        "NONE",
        "CONTEXT_ONLY",
    }:
        return 0.0


    # --------------------------------------------------------
    # 직접 인용
    # --------------------------------------------------------

    if evidence_type == "DIRECT_QUOTE":
        return 1.0


    # --------------------------------------------------------
    # Speaker에게 직접 귀속된 paraphrase
    # --------------------------------------------------------

    if (
        evidence_type == "ATTRIBUTED_PARAPHRASE"
        and directness == "DIRECT"
    ):

        if confidence == "HIGH":
            return 1.0

        return 0.80


    # --------------------------------------------------------
    # 기타 attributed paraphrase
    # --------------------------------------------------------

    if evidence_type == "ATTRIBUTED_PARAPHRASE":
        return 0.80


    # --------------------------------------------------------
    # 언론이 전달한 speaker position
    # 완전히 제거하지 않고 낮은 가중치 적용
    # --------------------------------------------------------

    if evidence_type == "REPORTED_POSITION":

        if policy_phrase:
            return 0.50


    return 0.0


# ============================================================
# OFFICIAL STRONG SIGNALS
# ============================================================

def build_official_signals():

    # --------------------------------------------------------
    # app.py Tab1과 동일하게
    # Official 날짜 metadata는 documents.json에서 가져옴
    # --------------------------------------------------------

    documents = load_json(
        RESULTS_DIR
        / "documents.json"
    )

    segments = load_json(
        RESULTS_DIR
        / "segments.json"
    )

    stance_results = load_json(
        RESULTS_DIR
        / "segment_stance.json"
    )


    # --------------------------------------------------------
    # Segment map
    # --------------------------------------------------------

    segment_map = {
        row["segment_id"]: row
        for row in segments
    }


    # --------------------------------------------------------
    # Document map
    #
    # segment의 document_url과
    # documents.json의 URL을 연결
    # --------------------------------------------------------

    document_map = {}

    for document in documents:

        url = (
            document.get("url")
            or document.get("document_url")
            or document.get("link")
        )

        if url:
            document_map[url] = document


    signals = []


    for stance in stance_results:

        score = stance.get("score")

        if score is None:
            continue


        try:
            raw_score = float(score)

        except (TypeError, ValueError):
            continue


        # ----------------------------------------------------
        # Strong threshold
        # ----------------------------------------------------

        if abs(raw_score) < STRONG_THRESHOLD:
            continue


        # ----------------------------------------------------
        # Policy relevant
        # ----------------------------------------------------

        if not stance.get(
            "policy_relevant"
        ):
            continue


        # ----------------------------------------------------
        # Hawk / Dove만
        # ----------------------------------------------------

        if stance.get("stance") not in {
            "HAWKISH",
            "DOVISH",
        }:
            continue


        # ----------------------------------------------------
        # Evidence 필요
        # ----------------------------------------------------

        if not stance.get("evidence"):
            continue


        # ----------------------------------------------------
        # Segment
        # ----------------------------------------------------

        segment = segment_map.get(
            stance.get("segment_id")
        )

        if not segment:
            continue


        url = segment.get(
            "document_url",
            "",
        )


        # ----------------------------------------------------
        # Document metadata
        # ----------------------------------------------------

        document = document_map.get(
            url,
            {},
        )


        # ----------------------------------------------------
        # Official Date
        #
        # app.py와 같은 우선순위:
        #
        # 1. published_at
        # 2. date
        # 3. published
        # 4. URL YYYYMMDD fallback
        # ----------------------------------------------------

        raw_date = (
            document.get("published_at")
            or document.get("date")
            or document.get("published")
        )


        if raw_date:

            event_date = str(
                raw_date
            )[:10]

        else:

            event_date = extract_official_date(
                url
            )


        if not event_date:
            continue


        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        signals.append(
            {
                "date":
                    event_date,

                "speaker":
                    segment.get("speaker"),

                "source_type":
                    "OFFICIAL",

                "source":
                    document.get(
                        "source",
                        "Federal Reserve",
                    ),

                "raw_score":
                    raw_score,

                "quality_weight":
                    1.0,

                "weighted_score":
                    raw_score,

                # 기존 downstream 호환
                "score":
                    raw_score,

                "stance":
                    stance.get("stance"),

                "evidence":
                    stance.get("evidence"),

                "title":
                    document.get("title"),

                "url":
                    url,

                "segment_id":
                    stance.get(
                        "segment_id"
                    ),

                "speaker_evidence_type":
                    "OFFICIAL",

                "directness":
                    stance.get(
                        "directness"
                    ),

                "signal_strength":
                    stance.get(
                        "signal_strength"
                    ),

                "evidence_confidence":
                    stance.get(
                        "evidence_confidence"
                    ),

                "text_sufficiency":
                    stance.get(
                        "text_sufficiency"
                    ),

                "policy_action":
                    stance.get(
                        "policy_action"
                    ),

                "stance_driver":
                    stance.get(
                        "stance_driver"
                    ),

                "policy_bearing_phrase":
                    stance.get(
                        "policy_bearing_phrase"
                    ),
            }
        )


    return signals


# ============================================================
# NEWS STRONG SIGNALS
# ============================================================

def build_news_signals():
    """
    news_history.json의 최신 backfill 결과를
    직접 사용한다.

    Strong 여부:
        abs(raw_score) >= 0.70

    최종 event score:
        raw_score * quality_weight
    """

    history = load_json(
        RESULTS_DIR
        / "news_history.json"
    )


    signals = []


    for row in history:

        # ----------------------------------------------------
        # Detailed schema가 없는 과거 row는 사용하지 않음
        # ----------------------------------------------------

        if (
            "speaker_evidence_type"
            not in row
        ):
            continue


        # ----------------------------------------------------
        # Raw score
        # ----------------------------------------------------

        try:
            raw_score = float(
                row.get("score") or 0
            )

        except (TypeError, ValueError):
            continue


        # ----------------------------------------------------
        # Strong 여부는 RAW score로 판단
        # ----------------------------------------------------

        if (
            abs(raw_score)
            < STRONG_THRESHOLD
        ):
            continue


        # ----------------------------------------------------
        # Policy relevant
        # ----------------------------------------------------

        if not row.get(
            "policy_relevant"
        ):
            continue


        # ----------------------------------------------------
        # Hawk / Dove만
        # ----------------------------------------------------

        if row.get("stance") not in {
            "HAWKISH",
            "DOVISH",
        }:
            continue


        # ----------------------------------------------------
        # Evidence 필요
        # ----------------------------------------------------

        if not row.get("evidence"):
            continue


        # ----------------------------------------------------
        # Evidence quality
        # ----------------------------------------------------

        weight = get_quality_weight(
            row
        )


        # NONE / CONTEXT_ONLY 등
        if weight <= 0:
            continue


        weighted_score = (
            raw_score
            * weight
        )


        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        published_at = row.get(
            "published_at"
        )

        event_date = (
            published_at[:10]
            if published_at
            else None
        )

        if not event_date:
            continue


        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        signals.append(
            {
                "date":
                    event_date,

                "speaker":
                    row.get("speaker"),

                "source_type":
                    "NEWS",

                "source":
                    row.get("source"),

                # 원래 LLM score
                "raw_score":
                    raw_score,

                # Evidence quality
                "quality_weight":
                    weight,

                # Market reaction에 사용할
                # quality-adjusted signal
                "weighted_score":
                    weighted_score,

                # 기존 코드 호환
                "score":
                    weighted_score,

                "stance":
                    row.get("stance"),

                "evidence":
                    row.get("evidence"),

                "title":
                    row.get("title"),

                "url":
                    row.get("url"),

                # 새 News pipeline의 stable Final Event ID.
                # legacy 호환을 위해 segment_id도 같은 ID를 유지한다.
                "event_id":
                    (
                        row.get("event_id")
                        or row.get("segment_id")
                    ),

                "segment_id":
                    (
                        row.get("event_id")
                        or row.get("segment_id")
                    ),

                "speaker_evidence_type":
                    row.get(
                        "speaker_evidence_type"
                    ),

                "directness":
                    row.get(
                        "directness"
                    ),

                "signal_strength":
                    row.get(
                        "signal_strength"
                    ),

                "evidence_confidence":
                    row.get(
                        "evidence_confidence"
                    ),

                "text_sufficiency":
                    row.get(
                        "text_sufficiency"
                    ),

                "policy_action":
                    row.get(
                        "policy_action"
                    ),

                "stance_driver":
                    row.get(
                        "stance_driver"
                    ),

                "policy_bearing_phrase":
                    row.get(
                        "policy_bearing_phrase"
                    ),
            }
        )


    return signals


# ============================================================
# BUILD ALL STRONG SIGNALS
# ============================================================

def build_strong_signals():

    official = build_official_signals()

    news = build_news_signals()


    signals = (
        official
        + news
    )


    # --------------------------------------------------------
    # 최신 날짜 우선
    # 같은 날짜면 강한 weighted signal 우선
    # --------------------------------------------------------

    signals.sort(
        key=lambda row: (
            row.get("date") or "",
            abs(
                row.get(
                    "weighted_score",
                    0,
                )
            ),
        ),
        reverse=True,
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            signals,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 80)
    print("BUILD STRONG POLICY SIGNALS")
    print("=" * 80)

    print(
        f"Official Strong Signals: "
        f"{len(official)}"
    )

    print(
        f"News Strong Signals: "
        f"{len(news)}"
    )

    print(
        f"Total Strong Signals: "
        f"{len(signals)}"
    )


    if news:

        print()

        print(
            "News quality weights:"
        )

        weights = {}

        for row in news:

            weight = row[
                "quality_weight"
            ]

            weights[weight] = (
                weights.get(
                    weight,
                    0,
                )
                + 1
            )

        for weight in sorted(
            weights,
            reverse=True,
        ):

            print(
                f"  {weight:.2f}"
                f" : "
                f"{weights[weight]}"
            )


    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print("=" * 80)


    return signals


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_strong_signals()