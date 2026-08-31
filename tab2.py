from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

EVENT_MARKET_PATH = (
    MARKET_DIR
    / "event_market_reactions.json"
)


FOMC_MARKET_REVIEW_PATH = (
    MARKET_DIR
    / "fomc_market_review.json"
)


FOMC_EVENT_SIGNAL_PATH = (
    MARKET_DIR
    / "fomc_event_signals.json"
)


# ============================================================
# STYLE
# ============================================================

TAB2_STYLE = """
<style>

/* ============================================================
   SECTION
   ============================================================ */

.market-section {
    margin-top: 24px;
}

.market-section-title {
    color: #263746;
    font-size: 1.08rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.market-section-subtitle {
    color: #8a97a6;
    font-size: 0.78rem;
    margin-bottom: 12px;
}


/* ============================================================
   HOW TO READ
   ============================================================ */

.market-guide {
    background: #f8fafc;
    border: 1px solid #e7ebef;
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 16px;
}

.market-guide-title {
    color: #263746;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 9px;
}

.market-guide-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.market-guide-item {
    color: #66717d;
    font-size: 0.72rem;
    line-height: 1.45;
}

.market-guide-item b {
    color: #263746;
}

.market-guide-note {
    color: #8a97a6;
    font-size: 0.68rem;
    margin-top: 9px;
    padding-top: 8px;
    border-top: 1px solid #e9edf1;
}


/* ============================================================
   EVENT CARD
   ============================================================ */

.market-event-card {
    background: #ffffff;
    border: 1px solid #e4e9ee;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 11px;
}

.market-fomc-card {
    border-left: 4px solid #263746;
}


/* ============================================================
   HEADER
   ============================================================ */

.market-event-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 11px;
}

.market-event-date {
    color: #263746;
    font-size: 0.94rem;
    font-weight: 800;
}

.market-event-meta {
    color: #8a97a6;
    font-size: 0.68rem;
    margin-top: 3px;
}

.market-event-right {
    text-align: right;
    white-space: nowrap;
}


/* ============================================================
   BADGES
   ============================================================ */

.market-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 0.66rem;
    font-weight: 800;
    margin-left: 3px;
}

.market-hawk {
    background: #fdecec;
    color: #c62828;
    border: 1px solid #f7c9c9;
}

.market-dove {
    background: #eaf7ee;
    color: #1f8a4c;
    border: 1px solid #c8ead3;
}

.market-neutral {
    background: #f1f3f5;
    color: #59636e;
    border: 1px solid #dfe3e7;
}

.market-aligned {
    background: #eaf7ee;
    color: #1f8a4c;
    border: 1px solid #c8ead3;
}

.market-diverged {
    background: #fdecec;
    color: #c62828;
    border: 1px solid #f7c9c9;
}

.market-mixed {
    background: #fff7e8;
    color: #a66b00;
    border: 1px solid #f2d8a4;
}

.market-no-data {
    background: #f4f5f6;
    color: #7b8794;
    border: 1px solid #e1e5e9;
}

.market-fomc-badge {
    background: #263746;
    color: #ffffff;
    border: 1px solid #263746;
}


/* ============================================================
   SIGNAL VS MARKET
   ============================================================ */

.signal-compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 11px;
}

.signal-box {
    background: #fafbfc;
    border: 1px solid #edf0f3;
    border-radius: 8px;
    padding: 9px 11px;
}

.signal-label {
    color: #8a97a6;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.signal-value {
    color: #263746;
    font-size: 0.82rem;
    font-weight: 800;
}

.signal-description {
    color: #7b8794;
    font-size: 0.66rem;
    margin-top: 3px;
}


/* ============================================================
   ALIGNMENT SUMMARY
   ============================================================ */

.alignment-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f8fafc;
    border-radius: 7px;
    padding: 7px 10px;
    margin-bottom: 11px;
}

.alignment-summary-main {
    font-size: 0.72rem;
    font-weight: 800;
    color: #263746;
}

.alignment-summary-detail {
    color: #7b8794;
    font-size: 0.66rem;
}


/* ============================================================
   SOURCE / EVIDENCE
   ============================================================ */

.evidence-section-title {
    color: #8a97a6;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 10px 0 5px 0;
}

.news-source {
    border-top: 1px solid #eef1f4;
    padding: 7px 0 6px 0;
}

.news-source:first-of-type {
    border-top: none;
}

.news-title {
    color: #263746;
    font-size: 0.72rem;
    font-weight: 750;
    line-height: 1.35;
}

.news-meta {
    color: #9aa4ae;
    font-size: 0.61rem;
    margin-top: 2px;
}

.news-evidence {
    color: #4b5563;
    font-size: 0.71rem;
    line-height: 1.48;
    margin-top: 4px;
}

.official-evidence {
    color: #4b5563;
    font-size: 0.71rem;
    line-height: 1.48;
    background: #fafbfc;
    border-radius: 7px;
    padding: 8px 10px;
}


/* ============================================================
   MARKET ROW
   ============================================================ */

.market-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-top: 7px;
}

.market-row-fomc {
    grid-template-columns: repeat(2, 1fr);
}

.market-mini {
    background: #fafbfc;
    border: 1px solid #edf0f3;
    border-radius: 7px;
    padding: 7px 5px;
    text-align: center;
}

.market-mini-name {
    color: #8a97a6;
    font-size: 0.60rem;
    font-weight: 700;
}

.market-mini-value {
    color: #263746;
    font-size: 0.79rem;
    font-weight: 800;
    margin-top: 2px;
}

.market-mini-status {
    font-size: 0.60rem;
    font-weight: 800;
    margin-top: 2px;
}

.status-aligned {
    color: #1f8a4c;
}

.status-diverged {
    color: #c62828;
}

.status-mixed {
    color: #a66b00;
}

.status-neutral {
    color: #8a97a6;
}

.status-hawkish {
    color: #c62828;
}

.status-dovish {
    color: #1f8a4c;
}


/* ============================================================
   FOMC
   ============================================================ */

.fomc-decision {
    display: flex;
    gap: 8px;
    align-items: center;
    background: #f7f9fb;
    border: 1px solid #e7ebef;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 9px;
}

.fomc-decision-label {
    color: #8a97a6;
    font-size: 0.62rem;
    font-weight: 800;
    text-transform: uppercase;
}

.fomc-decision-value {
    color: #263746;
    font-size: 0.78rem;
    font-weight: 800;
}

.fomc-validation {
    text-align: center;
    padding: 8px 10px;
    border-radius: 7px;
    margin: 8px 0 10px 0;
    font-size: 0.72rem;
    font-weight: 800;
}

.fomc-validation-aligned {
    background: #edf8f1;
    color: #1f8a4c;
}

.fomc-validation-diverged {
    background: #fdf0f0;
    color: #c62828;
}

.fomc-validation-mixed {
    background: #fff8e9;
    color: #a66b00;
}

.fomc-validation-neutral {
    background: #f4f5f6;
    color: #7b8794;
}


/* ============================================================
   SELECTOR / MARKET DATE META
   ============================================================ */

.market-date-note {
    color: #7b8794;
    font-size: 0.66rem;
    margin: -3px 0 9px 0;
}

.market-date-note b {
    color: #263746;
}

.fomc-calendar-title {
    color: #263746;
    font-size: 0.94rem;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 3px;
}

.fomc-calendar-subtitle {
    color: #8a97a6;
    font-size: 0.72rem;
    margin-bottom: 8px;
}

/* ============================================================
   STRONG SIGNAL DATE SELECTOR
   ============================================================ */

.strong-signal-title {
    color: #263746;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin: 2px 0 2px 0;
}

.strong-signal-subtitle {
    color: #8a97a6;
    font-size: 0.68rem;
    margin-bottom: 4px;
}

/* Strong signal 날짜 selector: 날짜 카드처럼 보이는 가로 스크롤 */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    gap: 8px !important;
    padding: 4px 2px 10px 2px !important;
    scrollbar-width: thin;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
    border: 1px solid #e4e9ee !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    padding: 8px 12px !important;
    min-width: 108px !important;
    justify-content: center !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    border-color: #263746 !important;
    background: #f7f9fb !important;
    box-shadow: inset 0 -3px 0 #263746 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
    color: #263746 !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* ============================================================
   FOMC CALENDAR GRID
   ============================================================ */

.fomc-calendar-shell {
    background: #ffffff;
    border: 1px solid #e4e9ee;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 4px 0 10px 0;
}

.fomc-calendar-year {
    color: #263746;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
}

.fomc-calendar-help {
    color: #8a97a6;
    font-size: 0.66rem;
    margin-bottom: 8px;
}

.fomc-detail-empty {
    background: #fafbfc;
    border: 1px dashed #dfe4e9;
    border-radius: 9px;
    padding: 18px 16px;
    color: #7b8794;
    font-size: 0.72rem;
    line-height: 1.5;
}

/* FOMC month buttons only: key wrapper generated by Streamlit */
div[data-testid="stButton"] > button {
    min-height: 58px;
    border-radius: 9px;
    font-size: 0.72rem;
    font-weight: 800;
    line-height: 1.2;
}

/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .market-guide-grid {
        grid-template-columns: 1fr;
    }

    .signal-compare {
        grid-template-columns: 1fr;
    }

    .market-row,
    .market-row-fomc {
        grid-template-columns: repeat(2, 1fr);
    }

}

</style>
"""


# ============================================================
# LOAD
# ============================================================

def load_event_market_reactions():

    if not EVENT_MARKET_PATH.exists():
        return []

    try:
        data = json.loads(
            EVENT_MARKET_PATH.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []

    if not isinstance(data, list):
        return []

    return data


def load_fomc_market_review():

    if not FOMC_MARKET_REVIEW_PATH.exists():
        return []

    try:
        data = json.loads(
            FOMC_MARKET_REVIEW_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []

    if not isinstance(data, list):
        return []

    return data


def load_fomc_event_signals():

    if not FOMC_EVENT_SIGNAL_PATH.exists():
        return []

    try:
        data = json.loads(
            FOMC_EVENT_SIGNAL_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []

    if not isinstance(data, list):
        return []

    return data


# ============================================================
# HTML
# ============================================================

def _render_html(html_text):

    clean_html = "".join(
        line.strip()
        for line in html_text.splitlines()
    )

    st.markdown(
        clean_html,
        unsafe_allow_html=True,
    )


def _escape(value):

    if value is None:
        return ""

    return html.escape(
        str(value)
    )


# ============================================================
# FORMAT
# ============================================================

def _format_score(value):

    if value is None:
        return "-"

    try:
        return f"{float(value):+.2f}"

    except (
        TypeError,
        ValueError,
    ):
        return "-"


def _format_bp(value):

    if value is None:
        return "-"

    try:
        return f"{float(value):+.1f}bp"

    except (
        TypeError,
        ValueError,
    ):
        return "-"


def _format_pct(value):

    if value is None:
        return "-"

    try:
        return f"{float(value):+.2f}%"

    except (
        TypeError,
        ValueError,
    ):
        return "-"


# ============================================================
# LABELS
# ============================================================

def _match_type_label(value):

    return {
        "OFFICIAL_NEWS_MATCHED":
            "Official + News",

        "OFFICIAL_ONLY":
            "Official",

        "NEWS_ONLY":
            "News",
    }.get(
        str(value or ""),
        str(value or "-"),
    )


def _stance_description(stance):

    stance = str(
        stance or ""
    ).upper()

    return {
        "HAWKISH":
            "긴축적 정책 신호",

        "DOVISH":
            "완화적 정책 신호",

        "NEUTRAL":
            "뚜렷한 정책 방향 없음",
    }.get(
        stance,
        "정책 방향 판정",
    )


def _alignment_description(alignment):

    alignment = str(
        alignment or ""
    ).upper()

    return {
        "ALIGNED":
            "시장 반응이 LLM 정책 신호와 같은 방향",

        "DIVERGED":
            "시장 반응이 LLM 정책 신호와 반대 방향",

        "MIXED":
            "자산별 시장 반응이 엇갈림",

        "NO_MARKET_DATA":
            "시장 데이터 없음",
    }.get(
        alignment,
        "시장 검증 결과",
    )


def _alignment_icon(value):

    value = str(
        value or ""
    ).upper()

    return {
        "ALIGNED": "✓",
        "DIVERGED": "✕",
        "MIXED": "△",
        "NEUTRAL": "–",
    }.get(
        value,
        "–",
    )


def _alignment_class(value):

    value = str(
        value or ""
    ).upper()

    return {
        "ALIGNED":
            "status-aligned",

        "DIVERGED":
            "status-diverged",

        "MIXED":
            "status-mixed",

        "NEUTRAL":
            "status-neutral",

        "HAWKISH":
            "status-hawkish",

        "DOVISH":
            "status-dovish",
    }.get(
        value,
        "status-neutral",
    )


def _badge(value):

    value = str(
        value or ""
    ).upper()

    css_class = {
        "HAWKISH":
            "market-hawk",

        "DOVISH":
            "market-dove",

        "NEUTRAL":
            "market-neutral",

        "ALIGNED":
            "market-aligned",

        "DIVERGED":
            "market-diverged",

        "MIXED":
            "market-mixed",

        "NO_MARKET_DATA":
            "market-no-data",
    }.get(
        value,
        "market-neutral",
    )

    return (
        f'<span class="market-badge {css_class}">'
        f'{_escape(value)}'
        f'</span>'
    )


# ============================================================
# GUIDE
# ============================================================

def render_market_guide():

    _render_html(
        """
        <div class="market-guide">

            <div class="market-guide-title">
                HOW TO READ
            </div>

            <div class="market-guide-grid">

                <div class="market-guide-item">
                    <b>Policy Signal</b><br>
                    우리 LLM이 Fed 발언을
                    Dovish ↔ Neutral ↔ Hawkish로 판정합니다.
                    +1에 가까울수록 Hawkish,
                    -1에 가까울수록 Dovish입니다.
                </div>

                <div class="market-guide-item">
                    <b>Market Validation</b><br>
                    발언 이후 실제 금리·달러·주식 등의 움직임이
                    LLM이 판정한 정책 방향과 일치했는지 비교합니다.
                </div>

                <div class="market-guide-item">
                    <b>✓ Aligned / ✕ Diverged</b><br>
                    Aligned는 시장 상승을 의미하지 않습니다.
                    LLM 정책 신호와 시장 해석의 방향이
                    같았다는 의미입니다.
                </div>

            </div>

            <div class="market-guide-note">
                FOMC 날짜는 일반 Daily Market 이벤트와 분리하여
                SF Fed USMPD의 intraday monetary-event signal로 검증합니다.
            </div>

        </div>
        """
    )


# ============================================================
# SCORE SUMMARY
# ============================================================

def _score_summary(event):

    if (
        event.get("match_type")
        != "OFFICIAL_NEWS_MATCHED"
    ):
        return ""

    return f"""
    <div class="alignment-summary">

        <div class="alignment-summary-main">
            Source scores
        </div>

        <div class="alignment-summary-detail">
            Official
            {_format_score(event.get("official_score"))}
            &nbsp;&nbsp;·&nbsp;&nbsp;

            News
            {_format_score(event.get("news_score"))}
            &nbsp;&nbsp;·&nbsp;&nbsp;

            Combined
            {_format_score(event.get("combined_score"))}
        </div>

    </div>
    """


# ============================================================
# NEWS SOURCES
# ============================================================

def _news_sources_html(event, max_sources=3):

    sources = (
        event.get("news_sources")
        or []
    )

    if not sources:

        evidence = (
            event.get("news_evidence")
            or event.get("key_evidence")
        )

        if not evidence:
            return ""

        return f"""
        <div class="evidence-section-title">
            News Evidence
        </div>

        <div class="news-source">
            <div class="news-evidence">
                “{_escape(evidence)}”
            </div>
        </div>
        """

    blocks = []

    for source in sources[:max_sources]:

        title = (
            source.get("title")
            or "News source"
        )

        source_name = (
            source.get("source")
            or ""
        )

        evidence = (
            source.get("evidence")
            or source.get("policy_bearing_phrase")
            or "-"
        )

        blocks.append(
            f"""
            <div class="news-source">

                <div class="news-title">
                    {_escape(title)}
                </div>

                <div class="news-meta">
                    {_escape(source_name)}
                </div>

                <div class="news-evidence">
                    “{_escape(evidence)}”
                </div>

            </div>
            """
        )

    return (
        '<div class="evidence-section-title">'
        'News Evidence'
        '</div>'
        + "".join(blocks)
    )


def _official_evidence_html(event):

    evidence = event.get(
        "official_evidence"
    )

    if not evidence:
        return ""

    title = (
        event.get("official_title")
        or "Official statement"
    )

    return f"""
    <div class="evidence-section-title">
        Official Evidence
    </div>

    <div class="official-evidence">
        <b>{_escape(title)}</b><br>
        “{_escape(evidence)}”
    </div>
    """


# ============================================================
# MARKET CELLS
# ============================================================

def _daily_market_cell(
    name,
    value,
    formatter,
    status,
):

    status_text = (
        str(status).upper()
        if status
        else "-"
    )

    return f"""
    <div class="market-mini">

        <div class="market-mini-name">
            {_escape(name)}
        </div>

        <div class="market-mini-value">
            {formatter(value)}
        </div>

        <div class="
            market-mini-status
            {_alignment_class(status_text)}
        ">
            {_alignment_icon(status_text)}
            {_escape(status_text)}
        </div>

    </div>
    """


def _fomc_market_cell(
    name,
    value,
    formatter,
    signal,
):

    signal = str(
        signal or "-"
    ).upper()

    return f"""
    <div class="market-mini">

        <div class="market-mini-name">
            {_escape(name)}
        </div>

        <div class="market-mini-value">
            {formatter(value)}
        </div>

        <div class="
            market-mini-status
            {_alignment_class(signal)}
        ">
            {_escape(signal)}
        </div>

    </div>
    """


# ============================================================
# DAILY EVENT CARD
# ============================================================

def render_daily_market_card(event):

    date_value = event.get(
        "date",
        "-"
    )

    stance = str(
        event.get(
            "stance",
            "UNKNOWN"
        )
    ).upper()

    combined_score = event.get(
        "combined_score"
    )

    alignment = str(
        event.get(
            "overall_alignment",
            "NO_MARKET_DATA"
        )
    ).upper()

    match_type = _match_type_label(
        event.get(
            "match_type"
        )
    )

    alignment_map = (
        event.get("alignment")
        or {}
    )

    aligned_count = event.get(
        "aligned_count"
    )

    available_markets = event.get(
        "available_markets"
    )

    alignment_ratio = event.get(
        "alignment_ratio"
    )

    if (
        alignment_ratio
        is not None
    ):
        try:
            ratio_text = (
                f"{float(alignment_ratio) * 100:.0f}%"
            )
        except (
            TypeError,
            ValueError,
        ):
            ratio_text = "-"
    else:
        ratio_text = "-"

    if (
        aligned_count is not None
        and available_markets
    ):
        alignment_count_text = (
            f"{aligned_count} / "
            f"{available_markets} markets aligned"
        )
    else:
        alignment_count_text = (
            "Market comparison unavailable"
        )

    html_block = f"""
    <div class="market-event-card">

        <div class="market-event-header">

            <div>

                <div class="market-event-date">
                    {_escape(date_value)}
                </div>

                <div class="market-event-meta">
                    {_escape(match_type)}
                    &nbsp;·&nbsp;
                    Daily Market
                </div>

            </div>

            <div class="market-event-right">
                {_badge(stance)}
                <span style="
                    font-size:0.76rem;
                    font-weight:800;
                    color:#263746;
                    margin-left:4px;
                ">
                    {_format_score(combined_score)}
                </span>
            </div>

        </div>


        <div class="market-date-note">
            Event date <b>{_escape(date_value)}</b>
            &nbsp;·&nbsp; Market reaction <b>{_escape(event.get("market_date") or "-")}</b>
            {"&nbsp;·&nbsp; " + _escape(str(event.get("market_date_shift_reason") or "").replace("_", " ").title()) if event.get("market_date_shifted") else ""}
        </div>


        <div class="signal-compare">

            <div class="signal-box">

                <div class="signal-label">
                    LLM Policy Signal
                </div>

                <div class="signal-value">
                    {_escape(stance)}
                    &nbsp;
                    {_format_score(combined_score)}
                </div>

                <div class="signal-description">
                    {_escape(
                        _stance_description(
                            stance
                        )
                    )}
                </div>

            </div>


            <div class="signal-box">

                <div class="signal-label">
                    Market Validation
                </div>

                <div class="signal-value">
                    {_alignment_icon(alignment)}
                    {_escape(alignment)}
                </div>

                <div class="signal-description">
                    {_escape(
                        _alignment_description(
                            alignment
                        )
                    )}
                </div>

            </div>

        </div>


        <div class="alignment-summary">

            <div class="alignment-summary-main">
                {_escape(alignment_count_text)}
            </div>

            <div class="alignment-summary-detail">
                LLM ↔ Market agreement
                &nbsp;
                <b>{_escape(ratio_text)}</b>
            </div>

        </div>


        {_score_summary(event)}

        {_official_evidence_html(event)}

        {_news_sources_html(event)}


        <div class="evidence-section-title">
            Market Reaction
        </div>

        <div class="market-row">

            {_daily_market_cell(
                "UST 2Y",
                event.get(
                    "ust2y_change_bp"
                ),
                _format_bp,
                alignment_map.get(
                    "ust2y"
                ),
            )}

            {_daily_market_cell(
                "UST 10Y",
                event.get(
                    "ust10y_change_bp"
                ),
                _format_bp,
                alignment_map.get(
                    "ust10y"
                ),
            )}

            {_daily_market_cell(
                "Broad USD",
                event.get(
                    "broad_usd_change_pct"
                ),
                _format_pct,
                alignment_map.get(
                    "broad_usd"
                ),
            )}

            {_daily_market_cell(
                "NASDAQ",
                event.get(
                    "nasdaq_change_pct"
                ),
                _format_pct,
                alignment_map.get(
                    "nasdaq"
                ),
            )}

            {_daily_market_cell(
                "USD/KRW",
                event.get(
                    "usdkrw_change_pct"
                ),
                _format_pct,
                alignment_map.get(
                    "usdkrw"
                ),
            )}

        </div>

    </div>
    """

    _render_html(
        html_block
    )



def _first_value(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_fomc_final_signal(item):
    if not isinstance(item, dict):
        return {}

    stance = _first_value(
        item, "final_stance", "stance", "llm_stance", "policy_stance"
    )
    score = _first_value(
        item, "final_score", "combined_score", "score", "llm_score", "policy_score"
    )
    action = _first_value(
        item, "action", "policy_action", "fomc_action", "rate_action"
    )
    bp_change = _first_value(
        item, "bp_change", "rate_change_bp", "change_bp", "decision_bp"
    )
    target_range = _first_value(
        item, "target_range", "target_rate", "target_rate_range"
    )

    lower = _first_value(
        item, "target_lower", "target_low", "target_lower_bound", "lower_bound"
    )
    upper = _first_value(
        item, "target_upper", "target_high", "target_upper_bound", "upper_bound"
    )

    target = item.get("target") or {}
    if isinstance(target, dict):
        if lower is None:
            lower = _first_value(target, "lower", "low", "lower_bound")
        if upper is None:
            upper = _first_value(target, "upper", "high", "upper_bound")

    if not target_range and lower is not None and upper is not None:
        try:
            target_range = f"{float(lower):.3f}% - {float(upper):.3f}%"
        except (TypeError, ValueError):
            target_range = f"{lower} - {upper}"

    final_label_ko = _first_value(
        item, "final_label_ko", "korean_label", "final_label_korean"
    )
    final_label_en = _first_value(
        item, "final_label", "final_label_en", "label"
    )

    if not stance and final_label_en:
        label = str(final_label_en).upper()
        if "HAWKISH" in label:
            stance = "HAWKISH"
        elif "DOVISH" in label:
            stance = "DOVISH"
        elif "NEUTRAL" in label:
            stance = "NEUTRAL"

    return {
        "stance": str(stance or "UNKNOWN").upper(),
        "score": score,
        "action": str(action).upper() if action else None,
        "bp_change": bp_change,
        "target_range": target_range,
        "final_label_ko": final_label_ko,
        "final_label_en": final_label_en,
    }


def _fomc_decision_text(event):
    action = event.get("fomc_action")
    bp_change = event.get("bp_change")
    target_range = event.get("target_range")

    if not action:
        return "Decision data not yet linked"

    parts = [str(action).upper()]

    # HOLD는 bp 표기를 생략한다.
    if str(action).upper() != "HOLD" and bp_change is not None:
        try:
            parts.append(f"{float(bp_change):+.0f}bp")
        except (TypeError, ValueError):
            parts.append(str(bp_change))

    if target_range:
        parts.append(str(target_range))

    return " · ".join(parts)


# ============================================================
# FOMC EVENT CARD
# ============================================================

def render_fomc_card(event):

    date_value = event.get(
        "date",
        "-"
    )

    stance = str(
        event.get(
            "stance",
            "UNKNOWN"
        )
    ).upper()

    combined_score = event.get(
        "combined_score"
    )

    match_type = _match_type_label(
        event.get(
            "match_type"
        )
    )

    overall_alignment = str(
        event.get(
            "overall_alignment",
            "NO_MARKET_DATA"
        )
    ).upper()

    market_interpretation = str(
        event.get(
            "fomc_market_interpretation",
            "UNKNOWN"
        )
    ).upper()

    market_strength = str(
        event.get(
            "fomc_market_strength",
            "-"
        )
    ).upper()

    usmpd_market = (
        event.get("usmpd_market")
        or {}
    )

    usmpd_signals = (
        event.get("usmpd_signals")
        or {}
    )

    # Final FOMC decision from fomc_event_signals.json.
    decision_text = _fomc_decision_text(event)

    # --------------------------------------------------------
    # FOMC validation summary
    # --------------------------------------------------------

    if overall_alignment == "ALIGNED":

        validation_class = (
            "fomc-validation-aligned"
        )

        validation_text = (
            "✓ LLM ↔ MARKET ALIGNED"
        )

    elif overall_alignment == "DIVERGED":

        validation_class = (
            "fomc-validation-diverged"
        )

        validation_text = (
            "✕ LLM ↔ MARKET DIVERGED"
        )

    elif overall_alignment == "MIXED":

        validation_class = (
            "fomc-validation-mixed"
        )

        validation_text = (
            "△ LLM ↔ MARKET MIXED"
        )

    else:

        validation_class = (
            "fomc-validation-neutral"
        )

        validation_text = (
            "LLM ↔ MARKET VALIDATION"
        )

    html_block = f"""
    <div class="
        market-event-card
        market-fomc-card
    ">

        <div class="market-event-header">

            <div>

                <div class="market-event-date">
                    {_escape(date_value)}
                    <span class="
                        market-badge
                        market-fomc-badge
                    ">
                        FOMC EVENT
                    </span>
                </div>

                <div class="market-event-meta">
                    {_escape(match_type)}
                    &nbsp;·&nbsp;
                    SF Fed USMPD Intraday
                </div>

            </div>

            <div class="market-event-right">

                {_badge(stance)}

                <span style="
                    font-size:0.76rem;
                    font-weight:800;
                    color:#263746;
                    margin-left:4px;
                ">
                    {_format_score(combined_score)}
                </span>

            </div>

        </div>


        <div class="fomc-decision">

            <div class="fomc-decision-label">
                FOMC Decision
            </div>

            <div class="fomc-decision-value">
                {_escape(decision_text)}
            </div>

        </div>


        <div class="signal-compare">

            <div class="signal-box">

                <div class="signal-label">
                    LLM Policy Signal
                </div>

                <div class="signal-value">
                    {_escape(event.get("final_label_ko") or stance)}
                    &nbsp;·&nbsp;
                    {_escape(stance)}
                    &nbsp;
                    {_format_score(combined_score)}
                </div>

                <div class="signal-description">
                    {_escape(
                        _stance_description(
                            stance
                        )
                    )}
                </div>

            </div>


            <div class="signal-box">

                <div class="signal-label">
                    Market Interpretation
                </div>

                <div class="signal-value">
                    {_escape(
                        market_interpretation
                    )}
                    &nbsp;·&nbsp;
                    {_escape(
                        market_strength
                    )}
                </div>

                <div class="signal-description">
                    SF Fed USMPD가 FOMC 직후
                    시장 움직임을 해석한 방향
                </div>

            </div>

        </div>


        <div class="
            fomc-validation
            {validation_class}
        ">
            {_escape(validation_text)}
        </div>


        {_score_summary(event)}

        {_official_evidence_html(event)}

        {_news_sources_html(
            event,
            max_sources=3,
        )}


    </div>
    """

    _render_html(
        html_block
    )

    _render_html(
        """
        <div class="evidence-section-title">
            FOMC Market Reaction · Intraday
        </div>
        """
    )

    # 실제 Streamlit columns로 2열 x 2행 고정.
    fomc_market_items = [
        (
            "UST 2Y",
            usmpd_market.get("ust2y_change_bp"),
            _format_bp,
            usmpd_signals.get("ust2y"),
        ),
        (
            "UST 10Y",
            usmpd_market.get("ust10y_change_bp"),
            _format_bp,
            usmpd_signals.get("ust10y"),
        ),
        (
            "S&P 500",
            usmpd_market.get("sp500_change_pct"),
            _format_pct,
            usmpd_signals.get("sp500"),
        ),
        (
            "DXY",
            usmpd_market.get("dxy_change_pct"),
            _format_pct,
            usmpd_signals.get("dxy"),
        ),
    ]

    for row_start in range(0, 4, 2):
        row_cols = st.columns(2, gap="small")

        for offset, col in enumerate(row_cols):
            name, value, formatter, signal = (
                fomc_market_items[row_start + offset]
            )

            with col:
                _render_html(
                    _fomc_market_cell(
                        name,
                        value,
                        formatter,
                        signal,
                    )
                )


# ============================================================
# MAIN
# ============================================================

def render_policy_signal_market_reaction(
    selected_member,
):

    st.markdown(
        TAB2_STYLE,
        unsafe_allow_html=True,
    )

    _render_html(
        """
        <div class="market-section">
            <div class="market-section-title">
                Policy Signal × Market Reaction
            </div>
            <div class="market-section-subtitle">
                Fed communication signal과 실제 시장 반응을 event 단위로 비교합니다.
            </div>
        </div>
        """
    )

    render_market_guide()

    events = load_event_market_reactions()

    if not events:
        st.info(
            "event_market_reactions.json 데이터가 없습니다."
        )
        return


    # --------------------------------------------------------
    # 1. GENERAL POLICY EVENTS
    # selected_member의 일반 발언 이벤트만 표시한다.
    # FOMC 이벤트는 아래 FOMC Calendar에서 별도로 표시한다.
    # --------------------------------------------------------

    daily_events = [
        event
        for event in events
        if (
            str(
                event.get(
                    "validation_source",
                    ""
                )
            ).upper()
            != "USMPD_FOMC"
            and str(
                event.get(
                    "speaker",
                    ""
                )
            ).strip()
            == str(
                selected_member
                or ""
            ).strip()
        )
    ]

    daily_events = sorted(
        daily_events,
        key=lambda x: str(
            x.get(
                "date",
                ""
            )
        ),
        reverse=True,
    )

    if daily_events:
        # signal_strength 필드가 실제로 존재하면 STRONG만 표시한다.
        # 아직 해당 필드가 없는 기존 JSON은 전체 policy event를 그대로 사용한다.
        has_signal_strength = any(
            event.get("signal_strength") is not None
            for event in daily_events
        )

        if has_signal_strength:
            selector_events = [
                event
                for event in daily_events
                if str(event.get("signal_strength", "")).upper()
                == "STRONG"
            ]
        else:
            selector_events = daily_events

        _render_html(
            """
            <div class="strong-signal-title">
                STRONG POLICY SIGNALS
            </div>
            <div class="strong-signal-subtitle">
                강한 정책 신호가 포착된 날짜를 가로로 탐색합니다.
            </div>
            """
        )

        if not selector_events:
            st.caption(
                "STRONG policy signal이 없습니다."
            )
        else:
            # 같은 날짜 이벤트가 여러 개여도 내부 option은 index라 충돌하지 않는다.
            selected_daily_index = st.radio(
                "Strong policy signal date",
                options=list(range(len(selector_events))),
                format_func=lambda i: str(
                    selector_events[i].get("date", "-")
                ),
                horizontal=True,
                label_visibility="collapsed",
                key="market_event_date_all",
            )

            selected_daily_event = (
                selector_events[selected_daily_index]
            )

            # 카드 1개만 렌더링한다.
            render_daily_market_card(
                selected_daily_event
            )

    else:
        st.info(
            "일반 발언 Market Reaction 데이터가 없습니다."
        )

    # --------------------------------------------------------
    # 2. FOMC CALENDAR
    # 일반 발언 selector와 완전히 분리한다.
    # 왼쪽: 2026 FOMC 월별 캘린더
    # 오른쪽: 선택 회의 상세
    # --------------------------------------------------------

    # Join market review and FINAL FOMC event signal by meeting date.
    fomc_reviews = load_fomc_market_review()
    fomc_event_signals = load_fomc_event_signals()

    review_by_date = {
        str(item.get("date") or "")[:10].strip(): item
        for item in fomc_reviews
        if isinstance(item, dict) and item.get("date")
    }

    fomc_signal_by_date = {
        str(item.get("date") or "")[:10].strip():
            _normalize_fomc_final_signal(item)
        for item in fomc_event_signals
        if isinstance(item, dict) and item.get("date")
    }

    # Critical fix: use BOTH JSON files.
    # A FOMC signal is displayed even when market-review data is missing.
    all_fomc_dates = sorted(
        set(review_by_date.keys())
        | set(fomc_signal_by_date.keys())
    )

    fomc_by_date = {}

    for meeting_date in all_fomc_dates:
        review = review_by_date.get(meeting_date) or {}
        final_signal = fomc_signal_by_date.get(meeting_date) or {}

        interpretation_block = review.get("market_interpretation") or {}
        if not isinstance(interpretation_block, dict):
            interpretation_block = {}

        intraday = review.get("intraday_market") or {}
        if not isinstance(intraday, dict):
            intraday = {}

        monetary = intraday.get("monetary_event") or {}
        if not isinstance(monetary, dict):
            monetary = {}

        signals = interpretation_block.get("signals") or {}
        if not isinstance(signals, dict):
            signals = {}

        fomc_by_date[meeting_date] = {
            "date": meeting_date,

            # These five fields come from fomc_event_signals.json.
            "stance": final_signal.get("stance") or "UNKNOWN",
            "combined_score": final_signal.get("score"),
            "fomc_action": final_signal.get("action"),
            "bp_change": final_signal.get("bp_change"),
            "target_range": final_signal.get("target_range"),
            "final_label_ko": final_signal.get("final_label_ko"),
            "final_label_en": final_signal.get("final_label_en"),

            # Market reaction remains from fomc_market_review.json.
            "match_type": "FOMC",
            "overall_alignment": (
                review.get("overall_alignment")
                or "NO_MARKET_DATA"
            ),
            "fomc_market_interpretation": (
                interpretation_block.get("interpretation")
                or review.get("fomc_market_interpretation")
                or "UNKNOWN"
            ),
            "fomc_market_strength": (
                interpretation_block.get("strength")
                or review.get("fomc_market_strength")
                or "-"
            ),
            "official_title": review.get("official_title"),
            "official_evidence": review.get("official_evidence"),
            "news_sources": review.get("news_sources") or [],
            "news_evidence": review.get("news_evidence"),
            "key_evidence": review.get("key_evidence"),
            "usmpd_signals": signals,
            "usmpd_market": {
                "ust2y_change_bp": monetary.get("ust2y_change_bp"),
                "ust10y_change_bp": monetary.get("ust10y_change_bp"),
                "sp500_change_pct": monetary.get("sp500_change_pct"),
                "dxy_change_pct": monetary.get("dxy_change_pct"),
            },
        }

    # 2026 FOMC scheduled meeting end dates.
    # 데이터가 아직 연결되지 않은 미래 회의도 캘린더에는 표시한다.
    fomc_schedule = [
        ("JAN", "2026-01-28"),
        ("MAR", "2026-03-18"),
        ("APR", "2026-04-29"),
        ("JUN", "2026-06-17"),
        ("JUL", "2026-07-29"),
        ("SEP", "2026-09-16"),
        ("OCT", "2026-10-28"),
        ("DEC", "2026-12-09"),
    ]

    _render_html(
        """
        <div class="fomc-calendar-title">
            FOMC Calendar
        </div>
        <div class="fomc-calendar-subtitle">
            2026 FOMC 회의 월을 선택하면 우측에서 정책 결정 · LLM signal · USMPD intraday reaction을 확인합니다.
        </div>
        """
    )

    # 최초 진입 시 데이터가 연결된 가장 최근 회의를 선택한다.
    available_schedule_dates = [
        meeting_date
        for _, meeting_date in fomc_schedule
        if meeting_date in fomc_by_date
    ]

    default_fomc_date = (
        available_schedule_dates[-1]
        if available_schedule_dates
        else fomc_schedule[0][1]
    )

    session_key = "tab2_selected_fomc_date"

    if st.session_state.get(session_key) not in [
        meeting_date
        for _, meeting_date in fomc_schedule
    ]:
        st.session_state[session_key] = default_fomc_date

    calendar_col, detail_col = st.columns(
        [1.05, 1.95],
        gap="large",
    )

    with calendar_col:
        _render_html(
            """
            <div class="fomc-calendar-shell">
                <div class="fomc-calendar-year">2026 FOMC</div>
                <div class="fomc-calendar-help">
                    ■ 선택 회의 &nbsp;·&nbsp; ● 데이터 연결 &nbsp;·&nbsp; ○ 예정 / 미연결
                </div>
            </div>
            """
        )

        # 2열 × 4행의 compact calendar.
        for row_start in range(0, len(fomc_schedule), 2):
            row_cols = st.columns(2, gap="small")

            for offset, col in enumerate(row_cols):
                idx = row_start + offset

                if idx >= len(fomc_schedule):
                    continue

                month_label, meeting_date = fomc_schedule[idx]
                has_data = meeting_date in fomc_by_date
                is_selected = (
                    st.session_state[session_key]
                    == meeting_date
                )

                day_label = meeting_date[5:].replace("-", "/")
                marker = "●" if has_data else "○"
                button_label = (
                    f"{marker} {month_label}\n{day_label}"
                )

                with col:
                    if st.button(
                        button_label,
                        key=f"fomc_calendar_{meeting_date}",
                        use_container_width=True,
                        type=(
                            "primary"
                            if is_selected
                            else "secondary"
                        ),
                    ):
                        st.session_state[session_key] = meeting_date
                        st.rerun()

    with detail_col:
        selected_fomc_date = st.session_state[session_key]
        selected_fomc_event = fomc_by_date.get(
            selected_fomc_date
        )

        if selected_fomc_event is not None:
            render_fomc_card(
                selected_fomc_event
            )
        else:
            selected_month = next(
                month
                for month, meeting_date in fomc_schedule
                if meeting_date == selected_fomc_date
            )

            _render_html(
                f"""
                <div class="market-event-card market-fomc-card">
                    <div class="market-event-header">
                        <div>
                            <div class="market-event-date">
                                {_escape(selected_fomc_date)}
                                <span class="market-badge market-fomc-badge">
                                    FOMC EVENT
                                </span>
                            </div>
                            <div class="market-event-meta">
                                {_escape(selected_month)} · 2026 FOMC Calendar
                            </div>
                        </div>
                    </div>
                    <div class="fomc-detail-empty">
                        이 회의는 캘린더에는 예정되어 있지만 현재
                        <b>fomc_market_review.json</b>에 연결된 FOMC/USMPD 데이터가 없습니다.<br>
                        데이터가 연결되면 같은 우측 영역에 FOMC Decision, LLM Policy Signal,
                        Market Interpretation, USMPD Intraday Reaction이 자동으로 표시됩니다.
                    </div>
                </div>
                """
            )