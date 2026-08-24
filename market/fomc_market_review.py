import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

FOMC_CALENDAR_PATH = (
    MARKET_DIR
    / "fomc_calendar.json"
)

MARKET_HISTORY_PATH = (
    MARKET_DIR
    / "market_history.csv"
)

USMPD_PATH = (
    MARKET_DIR
    / "USMPD.xlsx"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "fomc_market_review.json"
)


# ============================================================
# SOURCE
# ============================================================

USMPD_PAGE_URL = (
    "https://www.frbsf.org/"
    "research-and-insights/"
    "data-and-indicators/"
    "us-monetary-policy-event-study-database/"
)

USMPD_SOURCE_NAME = (
    "Federal Reserve Bank of San Francisco "
    "U.S. Monetary Policy Event-Study Database (USMPD)"
)


# ============================================================
# MARKET INTERPRETATION CONFIG
# ============================================================

# 변화가 이보다 작으면 방향성이 없는 것으로 본다.

UST2Y_NEUTRAL_BP = 1.0
UST10Y_NEUTRAL_BP = 1.0

SP500_NEUTRAL_PCT = 0.10
DXY_NEUTRAL_PCT = 0.10


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# NUMBER
# ============================================================

def clean_number(value):

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
# DAILY MARKET HISTORY
# ============================================================

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


def get_daily_market_row(
    market_df,
    fomc_date,
):

    target = pd.to_datetime(
        fomc_date,
        errors="coerce",
    )

    if pd.isna(target):
        return None

    matched = market_df[
        market_df["date"] == target
    ]

    if matched.empty:
        return None

    return matched.iloc[-1]


def build_daily_market(
    market_row,
):

    if market_row is None:
        return None

    return {

        "ust2y":
            clean_number(
                market_row.get(
                    "ust2y"
                )
            ),

        "ust2y_change_bp":
            clean_number(
                market_row.get(
                    "ust2y_change_bp"
                )
            ),

        "ust10y":
            clean_number(
                market_row.get(
                    "ust10y"
                )
            ),

        "ust10y_change_bp":
            clean_number(
                market_row.get(
                    "ust10y_change_bp"
                )
            ),

        "broad_usd":
            clean_number(
                market_row.get(
                    "broad_usd"
                )
            ),

        "broad_usd_change_pct":
            clean_number(
                market_row.get(
                    "broad_usd_change_pct"
                )
            ),

        "nasdaq":
            clean_number(
                market_row.get(
                    "nasdaq"
                )
            ),

        "nasdaq_change_pct":
            clean_number(
                market_row.get(
                    "nasdaq_change_pct"
                )
            ),

        "usdkrw":
            clean_number(
                market_row.get(
                    "usdkrw"
                )
            ),

        "usdkrw_change_pct":
            clean_number(
                market_row.get(
                    "usdkrw_change_pct"
                )
            ),
    }


# ============================================================
# FIND USMPD EXCEL DOWNLOAD LINK
# ============================================================

def find_usmpd_excel_url():

    response = requests.get(
        USMPD_PAGE_URL,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for link in soup.find_all(
        "a",
        href=True,
    ):

        href = link["href"]

        text = link.get_text(
            " ",
            strip=True,
        ).lower()

        href_lower = href.lower()

        if (
            "usmpd" in text
            and (
                ".xlsx" in href_lower
                or ".xls" in href_lower
            )
        ):
            return href

        if (
            "data for usmpd" in text
            and (
                ".xlsx" in href_lower
                or ".xls" in href_lower
            )
        ):
            return href

    raise RuntimeError(
        "USMPD Excel download link "
        "was not found."
    )


# ============================================================
# DOWNLOAD USMPD
# ============================================================

def download_usmpd():

    print(
        "Finding latest USMPD Excel..."
    )

    excel_url = (
        find_usmpd_excel_url()
    )

    if excel_url.startswith("/"):

        excel_url = (
            "https://www.frbsf.org"
            + excel_url
        )

    elif not excel_url.startswith(
        "http"
    ):

        excel_url = (
            "https://www.frbsf.org/"
            + excel_url.lstrip("/")
        )

    print(
        f"USMPD URL: {excel_url}"
    )

    response = requests.get(
        excel_url,
        timeout=60,
    )

    response.raise_for_status()

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        USMPD_PATH,
        "wb",
    ) as f:

        f.write(
            response.content
        )

    print(
        f"Downloaded: {USMPD_PATH}"
    )

    return USMPD_PATH


# ============================================================
# NORMALIZE USMPD DATE
# ============================================================

def find_date_column(df):

    candidates = [
        "Date",
        "date",
        "DATE",
        "EventDate",
        "event_date",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    return None


def normalize_usmpd_sheet(df):

    df = df.copy()

    date_col = (
        find_date_column(df)
    )

    if not date_col:

        raise RuntimeError(
            "USMPD date column "
            "was not found."
        )

    df["event_date"] = (
        pd.to_datetime(
            df[date_col],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    return df


# ============================================================
# LOAD USMPD
# ============================================================

def load_usmpd():

    sheets = pd.read_excel(
        USMPD_PATH,
        sheet_name=None,
    )

    required = [
        "Statements",
        "Press Conferences",
        "Monetary Events",
    ]

    result = {}

    for sheet_name in required:

        if sheet_name not in sheets:

            raise RuntimeError(
                f"USMPD sheet not found: "
                f"{sheet_name}"
            )

        result[sheet_name] = (
            normalize_usmpd_sheet(
                sheets[sheet_name]
            )
        )

    return result


# ============================================================
# FIND USMPD EVENT
# ============================================================

def get_usmpd_row(
    df,
    fomc_date,
):

    matched = df[
        df["event_date"]
        == fomc_date
    ]

    if matched.empty:
        return None

    return matched.iloc[-1]


# ============================================================
# USMPD MARKET FIELDS
# ============================================================

def build_usmpd_market(
    row,
):

    if row is None:
        return None

    ust2y = clean_number(
        row.get(
            "UST2Y"
        )
    )

    ust10y = clean_number(
        row.get(
            "UST10Y"
        )
    )

    sp500 = clean_number(
        row.get(
            "SP500"
        )
    )

    spfut = clean_number(
        row.get(
            "SPFUT"
        )
    )

    dxy = clean_number(
        row.get(
            "DXY"
        )
    )

    eurusd = clean_number(
        row.get(
            "EURUSD"
        )
    )

    usdjpy = clean_number(
        row.get(
            "USDJPY"
        )
    )

    return {

        # ----------------------------------------------------
        # Treasury
        #
        # USMPD yield change는 percentage point.
        # Dashboard 사용을 위해 bp로도 변환.
        # ----------------------------------------------------

        "ust2y_change_pp":
            ust2y,

        "ust2y_change_bp":
            (
                ust2y * 100
                if ust2y is not None
                else None
            ),

        "ust10y_change_pp":
            ust10y,

        "ust10y_change_bp":
            (
                ust10y * 100
                if ust10y is not None
                else None
            ),


        # ----------------------------------------------------
        # Equity
        # ----------------------------------------------------

        "sp500_change_pct":
            sp500,

        "sp500_futures_change_pct":
            spfut,


        # ----------------------------------------------------
        # FX
        # ----------------------------------------------------

        "dxy_change_pct":
            dxy,

        "eurusd_change_pct":
            eurusd,

        "usdjpy_change_pct":
            usdjpy,
    }


# ============================================================
# BUILD INTRADAY MARKET
# ============================================================

def build_intraday_market(
    usmpd,
    fomc_date,
):

    statement_row = (
        get_usmpd_row(
            usmpd[
                "Statements"
            ],
            fomc_date,
        )
    )

    press_row = (
        get_usmpd_row(
            usmpd[
                "Press Conferences"
            ],
            fomc_date,
        )
    )

    monetary_row = (
        get_usmpd_row(
            usmpd[
                "Monetary Events"
            ],
            fomc_date,
        )
    )

    if (
        statement_row is None
        and press_row is None
        and monetary_row is None
    ):
        return None

    return {

        # FOMC statement 주변
        # 30-minute event window

        "statement":
            build_usmpd_market(
                statement_row
            ),

        # Chair press conference
        #
        # Chair 이름은 코드에 넣지 않는다.
        # Powell / Warsh 등 역사적으로
        # 동일 구조를 사용할 수 있다.

        "press_conference":
            build_usmpd_market(
                press_row
            ),

        # Statement + Press Conference
        #
        # 최종 FOMC Market Interpretation은
        # 이 Monetary Event를 기준으로 한다.

        "monetary_event":
            build_usmpd_market(
                monetary_row
            ),
    }


# ============================================================
# CLASSIFY MARKET DIRECTION
# ============================================================

def classify_direction(
    value,
    threshold,
    positive_is_hawkish=True,
):
    """
    개별 시장 움직임을

    HAWKISH
    DOVISH
    NEUTRAL

    중 하나로 분류한다.
    """

    if value is None:
        return None

    if abs(value) < threshold:
        return "NEUTRAL"

    if positive_is_hawkish:

        return (
            "HAWKISH"
            if value > 0
            else "DOVISH"
        )

    return (
        "DOVISH"
        if value > 0
        else "HAWKISH"
    )


# ============================================================
# FOMC MARKET INTERPRETATION
# ============================================================

def interpret_fomc_market(
    intraday_market,
):
    """
    SF Fed USMPD Monetary Event를 이용해
    FOMC 시장 해석을 생성한다.

    Primary
    -------
    UST 2Y

    Confirmation
    ------------
    S&P 500
    DXY

    Secondary
    ---------
    UST 10Y

    최종 분류
    ---------
    HAWKISH
    DOVISH
    MIXED
    NEUTRAL
    """

    if not intraday_market:
        return None

    monetary = (
        intraday_market.get(
            "monetary_event"
        )
    )

    if not monetary:
        return None


    # --------------------------------------------------------
    # UST 2Y
    #
    # 금리 상승 = Hawkish
    # 금리 하락 = Dovish
    # --------------------------------------------------------

    ust2y_signal = (
        classify_direction(
            monetary.get(
                "ust2y_change_bp"
            ),
            UST2Y_NEUTRAL_BP,
            positive_is_hawkish=True,
        )
    )


    # --------------------------------------------------------
    # UST 10Y
    # --------------------------------------------------------

    ust10y_signal = (
        classify_direction(
            monetary.get(
                "ust10y_change_bp"
            ),
            UST10Y_NEUTRAL_BP,
            positive_is_hawkish=True,
        )
    )


    # --------------------------------------------------------
    # S&P 500
    #
    # 주가 하락 = Hawkish
    # 주가 상승 = Dovish
    # --------------------------------------------------------

    sp500_signal = (
        classify_direction(
            monetary.get(
                "sp500_change_pct"
            ),
            SP500_NEUTRAL_PCT,
            positive_is_hawkish=False,
        )
    )


    # --------------------------------------------------------
    # DXY
    #
    # Dollar 상승 = Hawkish
    # Dollar 하락 = Dovish
    # --------------------------------------------------------

    dxy_signal = (
        classify_direction(
            monetary.get(
                "dxy_change_pct"
            ),
            DXY_NEUTRAL_PCT,
            positive_is_hawkish=True,
        )
    )


    # ========================================================
    # PRIMARY
    # ========================================================

    primary = (
        ust2y_signal
    )


    # ========================================================
    # CONFIRMATION
    # ========================================================

    confirmation_signals = [
        sp500_signal,
        dxy_signal,
    ]

    confirmations = [
        signal
        for signal
        in confirmation_signals
        if signal
        not in {
            None,
            "NEUTRAL",
        }
    ]


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    if primary in {
        "HAWKISH",
        "DOVISH",
    }:

        opposite = (
            "DOVISH"
            if primary == "HAWKISH"
            else "HAWKISH"
        )

        opposite_count = (
            confirmations.count(
                opposite
            )
        )

        # S&P500과 DXY가 모두
        # UST2Y와 반대 방향일 경우만
        # MIXED 처리한다.

        if opposite_count == 2:
            final = "MIXED"

        else:
            final = primary


    else:

        # ----------------------------------------------------
        # UST2Y가 NEUTRAL일 경우
        #
        # S&P500 + DXY + UST10Y로 판단
        # ----------------------------------------------------

        secondary_signals = [
            sp500_signal,
            dxy_signal,
            ust10y_signal,
        ]

        usable = [
            signal
            for signal
            in secondary_signals
            if signal
            not in {
                None,
                "NEUTRAL",
            }
        ]

        hawkish_count = (
            usable.count(
                "HAWKISH"
            )
        )

        dovish_count = (
            usable.count(
                "DOVISH"
            )
        )

        if (
            hawkish_count == 0
            and dovish_count == 0
        ):

            final = "NEUTRAL"

        elif (
            hawkish_count
            > dovish_count
        ):

            final = "HAWKISH"

        elif (
            dovish_count
            > hawkish_count
        ):

            final = "DOVISH"

        else:

            final = "MIXED"


    # ========================================================
    # STRENGTH
    # ========================================================

    directional = [
        ust2y_signal,
        ust10y_signal,
        sp500_signal,
        dxy_signal,
    ]

    aligned_count = sum(
        signal == final
        for signal
        in directional
    )

    if final in {
        "MIXED",
        "NEUTRAL",
    }:

        strength = None

    elif aligned_count >= 3:

        strength = "STRONG"

    else:

        strength = "MODERATE"


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "interpretation":
            final,

        "strength":
            strength,

        "primary_signal":
            primary,

        "signals": {

            "ust2y":
                ust2y_signal,

            "ust10y":
                ust10y_signal,

            "sp500":
                sp500_signal,

            "dxy":
                dxy_signal,
        },
    }


# ============================================================
# BUILD ONE FOMC REVIEW
# ============================================================

def build_fomc_review(
    meeting,
    market_df,
    usmpd,
):

    fomc_date = (
        meeting.get(
            "date"
        )
    )

    daily_row = (
        get_daily_market_row(
            market_df,
            fomc_date,
        )
    )

    daily_market = (
        build_daily_market(
            daily_row
        )
    )

    intraday_market = (
        build_intraday_market(
            usmpd,
            fomc_date,
        )
    )

    market_interpretation = (
        interpret_fomc_market(
            intraday_market
        )
    )

    result = dict(
        meeting
    )

    result.update(
        {

            "daily_market_available":
                daily_market
                is not None,

            "intraday_market_available":
                intraday_market
                is not None,

            "daily_market":
                daily_market,

            "intraday_market":
                intraday_market,

            "market_interpretation":
                market_interpretation,

            "intraday_source":
                (
                    USMPD_SOURCE_NAME
                    if intraday_market
                    is not None
                    else None
                ),

            "intraday_source_url":
                (
                    USMPD_PAGE_URL
                    if intraday_market
                    is not None
                    else None
                ),
        }
    )

    return result


# ============================================================
# FORMAT
# ============================================================

def format_number(
    value,
    decimals=1,
):

    if value is None:
        return "None"

    return (
        f"{value:.{decimals}f}"
    )


# ============================================================
# BUILD
# ============================================================

def build_fomc_market_review():

    print("=" * 80)

    print(
        "BUILD FOMC MARKET REVIEW"
    )

    print("=" * 80)

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # FOMC Calendar
    # --------------------------------------------------------

    meetings = load_json(
        FOMC_CALENDAR_PATH
    )


    # --------------------------------------------------------
    # Daily Market
    # --------------------------------------------------------

    market_df = (
        load_market_history()
    )


    # --------------------------------------------------------
    # Download latest SF Fed USMPD
    # --------------------------------------------------------

    download_usmpd()


    # --------------------------------------------------------
    # Load USMPD
    # --------------------------------------------------------

    usmpd = (
        load_usmpd()
    )


    # --------------------------------------------------------
    # Build Reviews
    # --------------------------------------------------------

    results = []

    for meeting in meetings:

        result = (
            build_fomc_review(
                meeting,
                market_df,
                usmpd,
            )
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    daily_count = sum(
        row[
            "daily_market_available"
        ]
        for row
        in results
    )

    intraday_count = sum(
        row[
            "intraday_market_available"
        ]
        for row
        in results
    )


    print(
        f"FOMC Meetings     : "
        f"{len(results)}"
    )

    print(
        f"Daily Market      : "
        f"{daily_count}"
    )

    print(
        f"USMPD Intraday    : "
        f"{intraday_count}"
    )

    print()


    # --------------------------------------------------------
    # Print Monetary Event + Interpretation
    # --------------------------------------------------------

    for row in results:

        intraday = (
            row.get(
                "intraday_market"
            )
        )

        monetary = (
            intraday.get(
                "monetary_event"
            )
            if intraday
            else None
        )

        interpretation = (
            row.get(
                "market_interpretation"
            )
        )

        if monetary:

            interpretation_text = (
                interpretation.get(
                    "interpretation"
                )
                if interpretation
                else None
            )

            strength_text = (
                interpretation.get(
                    "strength"
                )
                if interpretation
                else None
            )

            print(
                f"{row.get('date')} "
                f"| {row.get('status')} "
                f"| 2Y="
                f"{format_number(monetary.get('ust2y_change_bp'), 1)}bp "
                f"| 10Y="
                f"{format_number(monetary.get('ust10y_change_bp'), 1)}bp "
                f"| SP500="
                f"{format_number(monetary.get('sp500_change_pct'), 2)}% "
                f"| DXY="
                f"{format_number(monetary.get('dxy_change_pct'), 2)}% "
                f"| {interpretation_text} "
                f"| {strength_text}"
            )

        else:

            print(
                f"{row.get('date')} "
                f"| {row.get('status')} "
                f"| USMPD=None"
            )


    # --------------------------------------------------------
    # Interpretation Summary
    # --------------------------------------------------------

    interpretations = [
        row.get(
            "market_interpretation"
        )
        for row
        in results
        if row.get(
            "market_interpretation"
        )
    ]

    hawkish_count = sum(
        row.get(
            "interpretation"
        ) == "HAWKISH"
        for row
        in interpretations
    )

    dovish_count = sum(
        row.get(
            "interpretation"
        ) == "DOVISH"
        for row
        in interpretations
    )

    mixed_count = sum(
        row.get(
            "interpretation"
        ) == "MIXED"
        for row
        in interpretations
    )

    neutral_count = sum(
        row.get(
            "interpretation"
        ) == "NEUTRAL"
        for row
        in interpretations
    )

    print()

    print(
        "Market Interpretation:"
    )

    print(
        f"  Hawkish : "
        f"{hawkish_count}"
    )

    print(
        f"  Dovish  : "
        f"{dovish_count}"
    )

    print(
        f"  Mixed   : "
        f"{mixed_count}"
    )

    print(
        f"  Neutral : "
        f"{neutral_count}"
    )


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    print()

    print(
        "Intraday Source:"
    )

    print(
        USMPD_SOURCE_NAME
    )

    print(
        USMPD_PAGE_URL
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

    build_fomc_market_review()