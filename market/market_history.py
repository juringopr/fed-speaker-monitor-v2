from pathlib import Path
from datetime import date

import pandas as pd
from pandas_datareader import data as web


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "market_history.csv"
)


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2026-01-01"

FRED_SERIES = {
    "DGS2": "ust2y",
    "DGS10": "ust10y",
    "DTWEXBGS": "broad_usd",
    "DEXKOUS": "usdkrw",
    "NASDAQCOM": "nasdaq",
}


# ============================================================
# FETCH
# ============================================================

def fetch_market_history(
    start_date,
    end_date,
):
    """
    공식/공식계열 시장 데이터를
    FRED를 통해 가져온다.

    DGS2
        UST 2Y Constant Maturity
        Fed H.15

    DGS10
        UST 10Y Constant Maturity
        Fed H.15

    DTWEXBGS
        Nominal Broad U.S. Dollar Index
        Fed H.10

    DEXKOUS
        South Korean Won per U.S. Dollar
        Fed H.10

    NASDAQCOM
        NASDAQ Composite Index
        Nasdaq / FRED
    """

    df = web.DataReader(
        list(FRED_SERIES.keys()),
        "fred",
        start_date,
        end_date,
    )

    df = df.rename(
        columns=FRED_SERIES
    )

    df.index = pd.to_datetime(
        df.index
    )

    df.index.name = "date"

    return df


# ============================================================
# CALCULATE DAILY CHANGES
# ============================================================

def add_daily_changes(df):
    """
    각 series의 직전 유효 관측치 대비 일별 변화를 계산한다.

    휴장일/결측치가 중간에 있으면 해당 NaN 행은 건너뛰고
    다음 유효 관측치를 직전 유효 거래일과 비교한다.

    예:
        6/18 = 4.19
        6/19 = NaN
        6/22 = 4.24

        → 6/22 UST 2Y change = +5bp
    """

    df = df.copy()


    # --------------------------------------------------------
    # Treasury Yield
    #
    # FRED 값은 %
    # 0.01%p = 1bp
    # --------------------------------------------------------

    df["ust2y_change_bp"] = (
        df["ust2y"]
        - df["ust2y"].ffill().shift(1)
    ) * 100

    df["ust10y_change_bp"] = (
        df["ust10y"]
        - df["ust10y"].ffill().shift(1)
    ) * 100


    # --------------------------------------------------------
    # Broad Dollar
    # --------------------------------------------------------

    broad_usd_prev = (
        df["broad_usd"]
        .ffill()
        .shift(1)
    )

    df["broad_usd_change_pct"] = (
        (
            df["broad_usd"]
            / broad_usd_prev
        )
        - 1
    ) * 100


    # --------------------------------------------------------
    # USD/KRW
    # --------------------------------------------------------

    usdkrw_prev = (
        df["usdkrw"]
        .ffill()
        .shift(1)
    )

    df["usdkrw_change_pct"] = (
        (
            df["usdkrw"]
            / usdkrw_prev
        )
        - 1
    ) * 100


    # --------------------------------------------------------
    # NASDAQ Composite
    # --------------------------------------------------------

    nasdaq_prev = (
        df["nasdaq"]
        .ffill()
        .shift(1)
    )

    df["nasdaq_change_pct"] = (
        (
            df["nasdaq"]
            / nasdaq_prev
        )
        - 1
    ) * 100


    return df

# ============================================================
# BUILD
# ============================================================

def build_market_history():

    end_date = date.today().isoformat()

    print("=" * 80)
    print("BUILD OFFICIAL MARKET HISTORY")
    print("=" * 80)

    print(
        f"Period: {START_DATE} -> {end_date}"
    )

    print()

    print(
        "Source: Federal Reserve Board / FRED / Nasdaq"
    )

    print("  DGS2      : UST 2Y")
    print("  DGS10     : UST 10Y")
    print("  DTWEXBGS  : Broad USD")
    print("  DEXKOUS   : USD/KRW")
    print("  NASDAQCOM : NASDAQ Composite")

    print()


    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    df = fetch_market_history(
        START_DATE,
        end_date,
    )

    print(
        f"Downloaded rows: {len(df)}"
    )


    # --------------------------------------------------------
    # Daily changes
    # --------------------------------------------------------

    df = add_daily_changes(df)


    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    df = df[
        [
            "ust2y",
            "ust2y_change_bp",

            "ust10y",
            "ust10y_change_bp",

            "broad_usd",
            "broad_usd_change_pct",

            "nasdaq",
            "nasdaq_change_pct",

            "usdkrw",
            "usdkrw_change_pct",
        ]
    ]


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print(
        f"Rows : {len(df)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print("=" * 80)

    print()

    print("Latest observations:")

    print(
        df.tail(10).to_string()
    )

    return df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_market_history()