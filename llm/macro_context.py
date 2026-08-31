from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MACRO_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "US_macro_index.xlsx"
)


# ============================================================
# Stance driver → 관련 경제지표 키워드
#
# Infomax "표시" 컬럼 기준.
# 너무 넓게 잡지 않고 Fed stance 해석에 직접 필요한 지표 위주.
# ============================================================

INDICATOR_GROUPS = {

    "INFLATION": [
        "CPI",
        "소비자물가지수",
        "PCE",
        "PPI",
        "GDP 물가지수",
        "수입가격",
        "수출가격",
    ],

    "LABOR": [
        "비농업 고용자수",
        "실업률",
        "시간당 평균",
        "참가율",
        "JOLTs",
        "ADP 고용",
        "실업수당",
        "고용비용",
        "단위 노동 비용",
        "ISM 제조업 고용",
        "ISM 서비스업 고용",
    ],

    "GROWTH": [
        "GDP성장률",
        "소매판매",
        "산업생산",
        "ISM 제조업 PMI",
        "ISM 서비스업 PMI",
        "S&P 제조업 PMI",
        "S&P 서비스업 PMI",
        "S&P 종합 PMI",
        "내구재 주문",
        "공장 주문",
        "소비자 신뢰",
        "Consumer Sentiment",
    ],

    "FINANCIAL_CONDITIONS": [
        "주택 착공",
        "주택 판매",
        "주택가격",
        "NAHB",
    ],

    "INTEREST_RATES": [],
    "BALANCE_SHEET": [],
    "POLICY_FRAMEWORK": [],
    "OTHER": [],
}


# ============================================================
# 제외 이벤트
#
# 중요도 중/상이어도 실제 economic indicator가 아닌 이벤트.
# ============================================================

EXCLUDE_KEYWORDS = [
    "Fed ",
    "연설",
    "발언",
    "기자회견",
    "FOMC",
    "재무부",
    "국채 입찰",
]


# ============================================================
# Excel 로드
# ============================================================

def load_macro_history(
    path: Path | str = MACRO_FILE,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Macro data file not found: {path}"
        )

    sheets = pd.read_excel(
        path,
        sheet_name=None,
    )

    frames = []

    for sheet_name, df in sheets.items():

        if df.empty:
            continue

        df = df.copy()
        df["sheet"] = sheet_name

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 날짜 + 시간 → release_datetime
    # --------------------------------------------------------

    history["release_datetime"] = pd.to_datetime(
        history["날짜"].astype(str).str.strip()
        + " "
        + history["시간"].astype(str).str.strip(),
        errors="coerce",
    )

    history = history.dropna(
        subset=["release_datetime"]
    )

    history = history.sort_values(
        "release_datetime"
    )

    return history.reset_index(drop=True)


# ============================================================
# Economic indicator가 아닌 이벤트 제거
# ============================================================

def remove_non_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    mask = pd.Series(
        False,
        index=df.index,
    )

    indicator_names = (
        df["표시"]
        .fillna("")
        .astype(str)
    )

    for keyword in EXCLUDE_KEYWORDS:

        mask |= indicator_names.str.contains(
            keyword,
            case=False,
            regex=False,
        )

    return df.loc[~mask].copy()


# ============================================================
# Driver 기준 관련 지표 필터
# ============================================================

def filter_by_driver(
    df: pd.DataFrame,
    driver: str | None,
) -> pd.DataFrame:

    if df.empty:
        return df

    if not driver:
        return df

    driver = driver.upper()

    keywords = INDICATOR_GROUPS.get(
        driver,
        [],
    )

    # POLICY_FRAMEWORK 등은
    # 억지로 macro indicator와 연결하지 않는다.
    if not keywords:
        return pd.DataFrame(
            columns=df.columns
        )

    indicator_names = (
        df["표시"]
        .fillna("")
        .astype(str)
    )

    mask = pd.Series(
        False,
        index=df.index,
    )

    for keyword in keywords:

        mask |= indicator_names.str.contains(
            keyword,
            case=False,
            regex=False,
        )

    return df.loc[mask].copy()


# ============================================================
# Macro 숫자 변환
#
# 비교만을 위한 함수.
#
# 예:
# 3.4%   -> 3.4
# 44K    -> 44000
# -23K   -> -23000
# 7.359M -> 7359000
#
# 여기서는 Hawk/Dove 의미를 해석하지 않는다.
# ============================================================

def _parse_numeric(value):

    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")

    if not text:
        return None

    multiplier = 1.0

    if text.endswith("%"):
        text = text[:-1]

    elif text.upper().endswith("K"):
        multiplier = 1_000
        text = text[:-1]

    elif text.upper().endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]

    elif text.upper().endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]

    try:
        return float(text) * multiplier

    except ValueError:
        return None


# ============================================================
# 두 Macro 값 비교
#
# ABOVE  = left > right
# BELOW  = left < right
# INLINE = left == right
#
# 경제적 의미는 판단하지 않는다.
# ============================================================

def _compare_values(
    left,
    right,
) -> str:

    left_num = _parse_numeric(left)
    right_num = _parse_numeric(right)

    if left_num is None or right_num is None:
        return "N/A"

    if left_num > right_num:
        return "ABOVE"

    if left_num < right_num:
        return "BELOW"

    return "INLINE"


# ============================================================
# Surprise / Trend 계산
#
# vs_consensus
#   actual vs 예상값
#
# vs_previous
#   actual vs 이전값
#
# 주의:
# BELOW/ABOVE는 숫자의 방향일 뿐,
# Hawk/Dove 방향을 의미하지 않는다.
# ============================================================

def add_macro_comparisons(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    df["vs_consensus"] = df.apply(
        lambda row: _compare_values(
            row.get("실제값"),
            row.get("예상값"),
        ),
        axis=1,
    )

    df["vs_previous"] = df.apply(
        lambda row: _compare_values(
            row.get("실제값"),
            row.get("이전값"),
        ),
        axis=1,
    )

    return df


# ============================================================
# 발언시점 기준 Macro Context 조회
# ============================================================

def get_macro_context(
    as_of: str | datetime,
    driver: str | None = None,
    lookback_days: int = 90,
    max_items: int = 17,
    path: Path | str = MACRO_FILE,
) -> pd.DataFrame:

    history = load_macro_history(path)

    if history.empty:
        return history

    history = remove_non_indicators(history)

    as_of = pd.Timestamp(as_of)

        # Excel release_datetime은 timezone-naive이므로
    # 뉴스 ISO timestamp의 timezone도 제거해서 비교 기준을 맞춘다.
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)

    start = as_of - pd.Timedelta(
        days=lookback_days
    )

    # --------------------------------------------------------
    # Look-ahead 방지
    #
    # 기사/발언 시점 이전에 실제 발표된 데이터만 사용
    # --------------------------------------------------------

    history = history[
        (history["release_datetime"] <= as_of)
        & (history["release_datetime"] >= start)
    ].copy()

    if driver:
        history = filter_by_driver(
            history,
            driver,
        )

    if history.empty:
        return history

    # --------------------------------------------------------
    # 기사 날짜 기준 최근 3개 월만 사용
    #
    # 예:
    # 기사 2026-08-20
    # → 2026-08 / 2026-07 / 2026-06
    #
    # 현재 월 데이터도 기사/발언 시점 이전 발표만 사용
    # --------------------------------------------------------

    article_month = as_of.to_period("M")

    valid_months = [
        article_month,
        article_month - 1,
        article_month - 2,
    ]

    history["context_month"] = (
        history["release_datetime"]
        .dt.to_period("M")
    )

    history = history[
        history["context_month"].isin(
            valid_months
        )
    ].copy()

    if history.empty:
        return history

    # --------------------------------------------------------
    # 최근 발표부터 정렬
    # --------------------------------------------------------

    history = history.sort_values(
        "release_datetime",
        ascending=False,
    )

    # --------------------------------------------------------
    # 월별 최대 max_items개
    #
    # 기존:
    #   전체 history에서 head(max_items)
    #
    # 문제:
    #   8월 + 7월 데이터가 max_items를 채우면
    #   6월 데이터가 완전히 사라질 수 있음.
    #
    # 변경:
    #   각 월마다 최대 max_items개를 유지.
    #
    # 예:
    #   max_items=12
    #   → 8월 최대 12개
    #   → 7월 최대 12개
    #   → 6월 최대 12개
    # --------------------------------------------------------

    history = (
        history
        .groupby(
            "context_month",
            group_keys=False,
            sort=False,
        )
        .head(max_items)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 월 → 최신 발표 순으로 다시 정렬
    # --------------------------------------------------------

    history = history.sort_values(
        "release_datetime",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Surprise / Previous 비교
    # --------------------------------------------------------

    history = add_macro_comparisons(
        history
    )

    return history


# ============================================================
# LLM Prompt용 문자열 생성
# ============================================================

def format_macro_context(
    df: pd.DataFrame,
) -> str:

    if df.empty:
        return "No relevant macroeconomic context available."

    lines = []

    current_month = None

    for _, row in df.iterrows():

        release_datetime = row[
            "release_datetime"
        ]

        month = release_datetime.strftime(
            "%Y-%m"
        )

        # ----------------------------------------------------
        # 월 변경 시 heading 추가
        # ----------------------------------------------------

        if month != current_month:

            if lines:
                lines.append("")

            lines.append(
                f"[{month}]"
            )

            current_month = month

        release_date = release_datetime.strftime(
            "%Y-%m-%d"
        )

        indicator = str(
            row.get("표시", "")
        ).strip()

        actual = _display_value(
            row.get("실제값")
        )

        consensus = _display_value(
            row.get("예상값")
        )

        forecast = _display_value(
            row.get("Forecast")
        )

        previous = _display_value(
            row.get("이전값")
        )

        reference_period = _display_value(
            row.get("발표월")
        )

        vs_consensus = row.get(
            "vs_consensus",
            "N/A",
        )

        vs_previous = row.get(
            "vs_previous",
            "N/A",
        )

        lines.append(
            f"- {indicator} "
            f"(released {release_date}, "
            f"reference {reference_period}): "
            f"actual={actual}, "
            f"consensus={consensus}, "
            f"forecast={forecast}, "
            f"previous={previous}, "
            f"vs_consensus={vs_consensus}, "
            f"vs_previous={vs_previous}"
        )

    return "\n".join(lines)


# ============================================================
# Segment → LLM용 Macro Context
# ============================================================

def build_macro_context(
    as_of: str | datetime,
    driver: str | None,
    lookback_days: int = 90,
    max_items: int = 12,
    path: Path | str = MACRO_FILE,
) -> str:

    df = get_macro_context(
        as_of=as_of,
        driver=driver,
        lookback_days=lookback_days,
        max_items=max_items,
        path=path,
    )

    return format_macro_context(df)


# ============================================================
# 내부 helper
# ============================================================

def _display_value(value) -> str:

    if pd.isna(value):
        return "N/A"

    return str(value).strip()


# ============================================================
# 단독 테스트
# ============================================================

if __name__ == "__main__":

    test_date = "2026-08-20 23:59"

    for driver in [
        "INFLATION",
        "LABOR",
        "GROWTH",
        "POLICY_FRAMEWORK",
    ]:

        print("\n" + "=" * 80)
        print(driver)
        print("=" * 80)

        print(
            build_macro_context(
                as_of=test_date,
                driver=driver,
            )
        )