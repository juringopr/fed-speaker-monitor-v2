import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

EVENTS_PATH = (
    MARKET_DIR
    / "aggregated_events.json"
)

MARKET_HISTORY_PATH = (
    MARKET_DIR
    / "market_history.csv"
)

FOMC_MARKET_REVIEW_PATH = (
    MARKET_DIR
    / "fomc_market_review.json"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "event_market_reactions.json"
)


# ============================================================
# LOAD
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_market_history():

    df = pd.read_csv(
        MARKET_HISTORY_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FOMC MARKET REVIEW
# ============================================================

def load_fomc_market_review():
    """
    SF Fed USMPD 기반 FOMC Market Review를 날짜별로 로드한다.

    FOMC 날짜와 Policy Event 날짜가 같으면
    일반 Daily Market보다 이 데이터를 우선 사용한다.
    """

    if not FOMC_MARKET_REVIEW_PATH.exists():
        return {}

    rows = load_json(
        FOMC_MARKET_REVIEW_PATH
    )

    return {
        row.get("date"): row
        for row in rows
        if row.get("date")
    }


def stance_vs_market_interpretation(
    stance,
    market_interpretation,
):
    """
    LLM stance와 USMPD FOMC market interpretation을 비교한다.
    """

    if not market_interpretation:
        return "NO_MARKET_DATA"

    if market_interpretation in {
        "MIXED",
        "NEUTRAL",
    }:
        return "MIXED"

    if stance == market_interpretation:
        return "ALIGNED"

    if (
        stance in {"HAWKISH", "DOVISH"}
        and market_interpretation in {"HAWKISH", "DOVISH"}
    ):
        return "DIVERGED"

    return "MIXED"


def build_fomc_market_reaction(
    event,
    fomc_review,
):
    """
    FOMC 날짜에는 SF Fed USMPD Monetary Event를
    시장 움직임 검증의 우선 데이터로 사용한다.

    주의:
    이는 개별 발언이 시장 움직임을 '유발했다'는 뜻이 아니라,
    LLM이 읽은 정책 방향과 해당 FOMC에 대한 시장의
    고빈도 해석이 같은 방향인지 검증하는 용도다.
    """

    intraday = (
        fomc_review.get("intraday_market")
        or {}
    )

    monetary = (
        intraday.get("monetary_event")
        or {}
    )

    interpretation_block = (
        fomc_review.get("market_interpretation")
        or {}
    )

    market_interpretation = (
        interpretation_block.get("interpretation")
    )

    market_strength = (
        interpretation_block.get("strength")
    )

    market_signals = (
        interpretation_block.get("signals")
        or {}
    )

    overall_alignment = (
        stance_vs_market_interpretation(
            event.get("stance"),
            market_interpretation,
        )
    )

    return {
        "market_date":
            fomc_review.get("date"),

        "market_date_shifted":
            False,

        "market_date_shift_reason":
            "FOMC_SAME_DAY",

        # 기존 Daily Market 필드는 보존하되,
        # FOMC 검증에서는 USMPD 전용 필드를 별도로 사용한다.
        "ust2y":
            None,

        "ust2y_change_bp":
            clean_number(
                monetary.get("ust2y_change_bp")
            ),

        "ust10y":
            None,

        "ust10y_change_bp":
            clean_number(
                monetary.get("ust10y_change_bp")
            ),

        "broad_usd":
            None,

        "broad_usd_change_pct":
            None,

        "nasdaq":
            None,

        "nasdaq_change_pct":
            None,

        "usdkrw":
            None,

        "usdkrw_change_pct":
            None,

        # 기존 alignment 구조는 깨지지 않도록 유지.
        # FOMC의 실제 세부 판정은 usmpd_signals에 저장한다.
        "alignment": {
            "ust2y": None,
            "ust10y": None,
            "broad_usd": None,
            "nasdaq": None,
            "usdkrw": None,
        },

        "aligned_count":
            0,

        "diverged_count":
            0,

        "neutral_count":
            0,

        "available_markets":
            sum(
                value is not None
                for value in [
                    monetary.get("ust2y_change_bp"),
                    monetary.get("ust10y_change_bp"),
                    monetary.get("sp500_change_pct"),
                    monetary.get("dxy_change_pct"),
                ]
            ),

        "directional_markets":
            sum(
                value in {"HAWKISH", "DOVISH"}
                for value in market_signals.values()
            ),

        "alignment_ratio":
            None,

        "overall_alignment":
            overall_alignment,

        # ----------------------------------------------------
        # Validation source
        # ----------------------------------------------------

        "validation_source":
            "USMPD_FOMC",

        "fomc_market_interpretation":
            market_interpretation,

        "fomc_market_strength":
            market_strength,

        "fomc_primary_signal":
            interpretation_block.get(
                "primary_signal"
            ),

        "usmpd_signals":
            market_signals,

        "usmpd_market": {
            "ust2y_change_bp":
                clean_number(
                    monetary.get(
                        "ust2y_change_bp"
                    )
                ),

            "ust10y_change_bp":
                clean_number(
                    monetary.get(
                        "ust10y_change_bp"
                    )
                ),

            "sp500_change_pct":
                clean_number(
                    monetary.get(
                        "sp500_change_pct"
                    )
                ),

            "dxy_change_pct":
                clean_number(
                    monetary.get(
                        "dxy_change_pct"
                    )
                ),
        },

        "validation_note":
            (
                "FOMC date matched. "
                "LLM policy stance is validated against "
                "SF Fed USMPD Monetary Event interpretation."
            ),
    }


# ============================================================
# VALUE HELPER
# ============================================================

def clean_number(value):
    """
    NaN / None을 JSON-safe None으로 변환.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# MARKET ROW
# ============================================================

def get_market_row(
    market_df,
    event_date,
):
    """
    event_date 이후의 데이터 중 실제 market reaction 값이 존재하는
    첫 번째 시장일을 사용한다.

    주말/휴장일뿐 아니라, 날짜 행은 존재하지만 reaction 값이 모두
    NaN인 경우도 건너뛴다.
    """

    event_ts = pd.to_datetime(
        event_date,
        errors="coerce",
    )

    if pd.isna(event_ts):
        return None

    reaction_columns = [
        "ust2y_change_bp",
        "ust10y_change_bp",
        "broad_usd_change_pct",
        "nasdaq_change_pct",
        "usdkrw_change_pct",
    ]

    candidates = (
        market_df[
            market_df["date"] >= event_ts
        ]
        .sort_values("date")
    )

    for _, row in candidates.iterrows():

        has_reaction = any(
            column in row.index
            and pd.notna(row[column])
            for column in reaction_columns
        )

        if has_reaction:
            return row

    return None


# ============================================================
# ALIGNMENT
# ============================================================

def check_alignment(
    stance,
    change,
):
    """
    금리 / 달러 계열 방향 alignment.

    HAWKISH
        금리 상승 / USD 상승
        -> ALIGNED

    DOVISH
        금리 하락 / USD 하락
        -> ALIGNED

    change == 0
        -> NEUTRAL
    """

    if change is None:
        return None

    if change == 0:
        return "NEUTRAL"

    if stance == "HAWKISH":

        if change > 0:
            return "ALIGNED"

        return "DIVERGED"

    if stance == "DOVISH":

        if change < 0:
            return "ALIGNED"

        return "DIVERGED"

    return None


def check_equity_alignment(
    stance,
    change,
):
    """
    주식시장은 금리 / 달러 계열과 반대 방향으로 해석.

    HAWKISH
        NASDAQ 하락
        -> ALIGNED

    DOVISH
        NASDAQ 상승
        -> ALIGNED

    change == 0
        -> NEUTRAL
    """

    if change is None:
        return None

    if change == 0:
        return "NEUTRAL"

    if stance == "HAWKISH":

        if change < 0:
            return "ALIGNED"

        return "DIVERGED"

    if stance == "DOVISH":

        if change > 0:
            return "ALIGNED"

        return "DIVERGED"

    return None


# ============================================================
# MARKET REACTION
# ============================================================

def build_market_reaction(
    event,
    market_row,
):

    stance = event.get(
        "stance"
    )


    # --------------------------------------------------------
    # Market date shift metadata
    # --------------------------------------------------------

    event_date = pd.to_datetime(
        event.get("date"),
        errors="coerce",
    )

    market_date = pd.to_datetime(
        market_row.get("date"),
        errors="coerce",
    )

    market_date_shifted = False
    market_date_shift_reason = "SAME_DAY"

    if (
        pd.notna(event_date)
        and pd.notna(market_date)
        and market_date.date() != event_date.date()
    ):
        market_date_shifted = True

        if event_date.weekday() >= 5:
            market_date_shift_reason = "WEEKEND"
        else:
            market_date_shift_reason = "NO_USABLE_MARKET_REACTION"


    # --------------------------------------------------------
    # Treasury 2Y
    # --------------------------------------------------------

    ust2y = clean_number(
        market_row.get(
            "ust2y"
        )
    )

    ust2y_change_bp = clean_number(
        market_row.get(
            "ust2y_change_bp"
        )
    )


    # --------------------------------------------------------
    # Treasury 10Y
    # --------------------------------------------------------

    ust10y = clean_number(
        market_row.get(
            "ust10y"
        )
    )

    ust10y_change_bp = clean_number(
        market_row.get(
            "ust10y_change_bp"
        )
    )


    # --------------------------------------------------------
    # Broad USD
    # --------------------------------------------------------

    broad_usd = clean_number(
        market_row.get(
            "broad_usd"
        )
    )

    broad_usd_change_pct = clean_number(
        market_row.get(
            "broad_usd_change_pct"
        )
    )


    # --------------------------------------------------------
    # NASDAQ Composite
    # --------------------------------------------------------

    nasdaq = clean_number(
        market_row.get(
            "nasdaq"
        )
    )

    nasdaq_change_pct = clean_number(
        market_row.get(
            "nasdaq_change_pct"
        )
    )


    # --------------------------------------------------------
    # USD/KRW
    # --------------------------------------------------------

    usdkrw = clean_number(
        market_row.get(
            "usdkrw"
        )
    )

    usdkrw_change_pct = clean_number(
        market_row.get(
            "usdkrw_change_pct"
        )
    )


    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    alignments = {

        "ust2y":
            check_alignment(
                stance,
                ust2y_change_bp,
            ),

        "ust10y":
            check_alignment(
                stance,
                ust10y_change_bp,
            ),

        "broad_usd":
            check_alignment(
                stance,
                broad_usd_change_pct,
            ),

        "nasdaq":
            check_equity_alignment(
                stance,
                nasdaq_change_pct,
            ),

        "usdkrw":
            check_alignment(
                stance,
                usdkrw_change_pct,
            ),
    }


    # --------------------------------------------------------
    # Alignment summary
    # --------------------------------------------------------

    valid = [
        value
        for value in alignments.values()
        if value is not None
    ]

    aligned_count = sum(
        value == "ALIGNED"
        for value in valid
    )

    diverged_count = sum(
        value == "DIVERGED"
        for value in valid
    )

    neutral_count = sum(
        value == "NEUTRAL"
        for value in valid
    )


    # --------------------------------------------------------
    # Overall Alignment
    #
    # ALIGNED / DIVERGED만 방향성 판정의 분모로 사용.
    # NEUTRAL은 분모에서 제외한다.
    #
    # 70% 이상 일치 -> ALIGNED
    # 70% 이상 반대 -> DIVERGED
    # 그 외         -> MIXED
    # --------------------------------------------------------

    directional_count = (
        aligned_count
        + diverged_count
    )

    if directional_count == 0:

        if valid:
            overall_alignment = "MIXED"
        else:
            overall_alignment = "NO_MARKET_DATA"

        alignment_ratio = None

    else:

        aligned_ratio = (
            aligned_count
            / directional_count
        )

        diverged_ratio = (
            diverged_count
            / directional_count
        )

        alignment_ratio = (
            aligned_ratio
        )

        if aligned_ratio >= 0.70:

            overall_alignment = (
                "ALIGNED"
            )

        elif diverged_ratio >= 0.70:

            overall_alignment = (
                "DIVERGED"
            )

        else:

            overall_alignment = (
                "MIXED"
            )


    return {

        "validation_source":
            "DAILY_MARKET",

        "fomc_market_interpretation":
            None,

        "fomc_market_strength":
            None,

        "fomc_primary_signal":
            None,

        "usmpd_signals":
            None,

        "usmpd_market":
            None,

        "validation_note":
            None,

        "market_date":
            market_row[
                "date"
            ].strftime(
                "%Y-%m-%d"
            ),

        "market_date_shifted":
            market_date_shifted,

        "market_date_shift_reason":
            market_date_shift_reason,


        # ----------------------------------------------------
        # Market values
        # ----------------------------------------------------

        "ust2y":
            ust2y,

        "ust2y_change_bp":
            ust2y_change_bp,

        "ust10y":
            ust10y,

        "ust10y_change_bp":
            ust10y_change_bp,

        "broad_usd":
            broad_usd,

        "broad_usd_change_pct":
            broad_usd_change_pct,

        "nasdaq":
            nasdaq,

        "nasdaq_change_pct":
            nasdaq_change_pct,

        "usdkrw":
            usdkrw,

        "usdkrw_change_pct":
            usdkrw_change_pct,


        # ----------------------------------------------------
        # Alignment
        # ----------------------------------------------------

        "alignment":
            alignments,

        "aligned_count":
            aligned_count,

        "diverged_count":
            diverged_count,

        "neutral_count":
            neutral_count,

        "available_markets":
            len(valid),

        "directional_markets":
            directional_count,

        "alignment_ratio":
            alignment_ratio,

        "overall_alignment":
            overall_alignment,
    }


# ============================================================
# NO MARKET DATA
# ============================================================

def build_empty_market_reaction():

    return {

        "validation_source":
            None,

        "fomc_market_interpretation":
            None,

        "fomc_market_strength":
            None,

        "fomc_primary_signal":
            None,

        "usmpd_signals":
            None,

        "usmpd_market":
            None,

        "validation_note":
            None,

        "market_date":
            None,

        "market_date_shifted":
            None,

        "market_date_shift_reason":
            None,

        "ust2y":
            None,

        "ust2y_change_bp":
            None,

        "ust10y":
            None,

        "ust10y_change_bp":
            None,

        "broad_usd":
            None,

        "broad_usd_change_pct":
            None,

        "nasdaq":
            None,

        "nasdaq_change_pct":
            None,

        "usdkrw":
            None,

        "usdkrw_change_pct":
            None,

        "alignment": {

            "ust2y":
                None,

            "ust10y":
                None,

            "broad_usd":
                None,

            "nasdaq":
                None,

            "usdkrw":
                None,
        },

        "aligned_count":
            0,

        "diverged_count":
            0,

        "neutral_count":
            0,

        "available_markets":
            0,

        "directional_markets":
            0,

        "alignment_ratio":
            None,

        "overall_alignment":
            "NO_MARKET_DATA",
    }


# ============================================================
# MATCH EVENT
# ============================================================

def match_event(
    event,
    market_df,
    fomc_reviews,
):

    event_date = event.get(
        "date"
    )


    # ========================================================
    # MARKET SOURCE PRIORITY
    #
    # 1. FOMC date
    #    -> SF Fed USMPD intraday market interpretation
    #
    # 2. Normal date
    #    -> existing Daily Market
    # ========================================================

    fomc_review = (
        fomc_reviews.get(
            event_date
        )
    )

    if (
        fomc_review
        and fomc_review.get(
            "intraday_market_available"
        )
        and fomc_review.get(
            "market_interpretation"
        )
    ):

        reaction = (
            build_fomc_market_reaction(
                event,
                fomc_review,
            )
        )

    else:

        market_row = get_market_row(
            market_df,
            event_date,
        )

        if market_row is None:

            reaction = (
                build_empty_market_reaction()
            )

        else:

            reaction = (
                build_market_reaction(
                    event,
                    market_row,
                )
            )


    # ========================================================
    # EVENT METADATA
    # ========================================================

    result = {

        # ----------------------------------------------------
        # Event identity
        # ----------------------------------------------------

        "event_id":
            event.get(
                "event_id"
            ),

        "date":
            event_date,

        "speaker":
            event.get(
                "speaker"
            ),

        "stance":
            event.get(
                "stance"
            ),

        "match_type":
            event.get(
                "match_type"
            ),


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        "official_score":
            clean_number(
                event.get(
                    "official_score"
                )
            ),

        "news_score":
            clean_number(
                event.get(
                    "news_score"
                )
            ),

        "combined_score":
            clean_number(
                event.get(
                    "combined_score"
                )
            ),


        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        "official_count":
            event.get(
                "official_count",
                0,
            ),

        "news_count":
            event.get(
                "news_count",
                0,
            ),


        # ----------------------------------------------------
        # Event description
        # ----------------------------------------------------

        "event_summary":
            event.get(
                "event_summary"
            ),

        "key_evidence":
            event.get(
                "key_evidence"
            ),

        "official_evidence":
            event.get(
                "official_evidence"
            ),

        "news_evidence":
            event.get(
                "news_evidence"
            ),


        # ----------------------------------------------------
        # Official source
        # ----------------------------------------------------

        "official_title":
            event.get(
                "official_title"
            ),

        "official_url":
            event.get(
                "official_url"
            ),

        "official_segment_id":
            event.get(
                "official_segment_id"
            ),


        # ----------------------------------------------------
        # News sources
        # ----------------------------------------------------

        "news_sources":
            event.get(
                "news_sources",
                [],
            ),


        # ----------------------------------------------------
        # News metadata
        # ----------------------------------------------------

        "speaker_evidence_type":
            event.get(
                "speaker_evidence_type"
            ),

        "directness":
            event.get(
                "directness"
            ),

        "signal_strength":
            event.get(
                "signal_strength"
            ),

        "evidence_confidence":
            event.get(
                "evidence_confidence"
            ),

        "policy_action":
            event.get(
                "policy_action"
            ),

        "stance_driver":
            event.get(
                "stance_driver"
            ),
    }


    # ========================================================
    # MARKET DATA
    # ========================================================

    result.update(
        reaction
    )

    return result


# ============================================================
# BUILD
# ============================================================

def build_event_market_reactions():

    # --------------------------------------------------------
    # 실제 발언 단위로 aggregation 완료된 Event 사용
    # --------------------------------------------------------

    events = load_json(
        EVENTS_PATH
    )

    market_df = (
        load_market_history()
    )

    fomc_reviews = (
        load_fomc_market_review()
    )

    results = []


    for event in events:

        result = match_event(
            event,
            market_df,
            fomc_reviews,
        )

        results.append(
            result
        )


    # ========================================================
    # SORT
    #
    # 날짜 우선
    # 같은 날짜면 |Combined Score| 큰 순서
    # ========================================================

    results.sort(
        key=lambda row: (
            row.get(
                "date"
            )
            or "",
            abs(
                clean_number(
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
    # SAVE
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
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    with_market = sum(
        row.get(
            "overall_alignment"
        )
        != "NO_MARKET_DATA"
        for row in results
    )

    no_market = sum(
        row.get(
            "overall_alignment"
        )
        == "NO_MARKET_DATA"
        for row in results
    )

    aligned = sum(
        row.get(
            "overall_alignment"
        )
        == "ALIGNED"
        for row in results
    )

    mixed = sum(
        row.get(
            "overall_alignment"
        )
        == "MIXED"
        for row in results
    )

    diverged = sum(
        row.get(
            "overall_alignment"
        )
        == "DIVERGED"
        for row in results
    )

    fomc_usmpd = sum(
        row.get(
            "validation_source"
        )
        == "USMPD_FOMC"
        for row in results
    )


    # ========================================================
    # MATCH TYPE SUMMARY
    # ========================================================

    official_news = sum(
        row.get(
            "match_type"
        )
        == "OFFICIAL_NEWS_MATCHED"
        for row in results
    )

    official_only = sum(
        row.get(
            "match_type"
        )
        == "OFFICIAL_ONLY"
        for row in results
    )

    news_only = sum(
        row.get(
            "match_type"
        )
        == "NEWS_ONLY"
        for row in results
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("=" * 80)

    print(
        "BUILD AGGREGATED EVENT × MARKET REACTIONS"
    )

    print("=" * 80)


    print(
        f"Policy Events    : "
        f"{len(events)}"
    )

    print(
        f"Matched Market   : "
        f"{with_market}"
    )

    print(
        f"No Market Data   : "
        f"{no_market}"
    )

    print(
        f"FOMC USMPD       : "
        f"{fomc_usmpd}"
    )

    print()


    print(
        f"Official + News  : "
        f"{official_news}"
    )

    print(
        f"Official Only    : "
        f"{official_only}"
    )

    print(
        f"News Only        : "
        f"{news_only}"
    )

    print()


    print(
        f"Aligned          : "
        f"{aligned}"
    )

    print(
        f"Mixed            : "
        f"{mixed}"
    )

    print(
        f"Diverged         : "
        f"{diverged}"
    )

    print()


    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    build_event_market_reactions()