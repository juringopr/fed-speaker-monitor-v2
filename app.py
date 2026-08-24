from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

try:
    # 로컬: C:\juringo 에서 package 실행
    from fed_speaker_monitor_v2.tab2 import (
        render_policy_signal_market_reaction,
    )
except ModuleNotFoundError:
    # Streamlit Cloud: repository root에서 app.py 실행
    from tab2 import (
        render_policy_signal_market_reaction,
    )


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "data" / "results"

NEWS_TREND_START = "2026-01-01"
NEWS_RECENT_DAYS = 14
NEWS_CURRENT_SCORE_DAYS = 30

# Combined Score는 Official과 최근 30일 News의 단순 평균.
# 한쪽만 있으면 존재하는 score를 그대로 사용.
COMBINED_OFFICIAL_WEIGHT = 0.50
COMBINED_NEWS_WEIGHT = 0.50

st.set_page_config(
    page_title="Fed Speaker Monitor V2",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .fed-topline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 24px;
        margin-bottom: 12px;
    }

    .fed-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        color: #111827;
    }

    .fed-mini-metrics {
        text-align: right;
        color: #6b7280;
        font-size: 0.82rem;
        line-height: 1.7;
        white-space: nowrap;
        padding-top: 5px;
    }

    .fed-info-box {
        background: #f3f7fc;
        border: 1px solid #e5edf7;
        border-radius: 10px;
        padding: 15px 18px;
        margin: 10px 0 20px 0;
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.65;
    }

    .fed-info-icon {
        display: inline-flex;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: #e4efff;
        color: #2563eb;
        font-weight: 800;
        margin-right: 8px;
    }

    .fed-table-wrap {
        overflow-x: auto;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: white;
    }

    table.fed-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.91rem;
    }

    .fed-table thead th {
        text-align: left;
        padding: 6px 14px;
        color: #475569;
        font-weight: 700;
        border-bottom: 1px solid #dfe5ec;
        background: #fafbfc;
        white-space: nowrap;
    }

    .fed-table tbody tr {
        height: 32px;
    }

    .fed-table tbody td {
        height: 32px;
        padding: 2px 14px;
        border-bottom: 1px solid #edf0f3;
        color: #1f2937;
        vertical-align: middle;
        line-height: 1.15;
    }

    .fed-table tbody tr:last-child td {
        border-bottom: none;
    }

    .fed-table tbody tr.voter-row td {
        font-weight: 700;
    }

    .voter-badge {
        display: inline-block;
        background: #e8f1ff;
        color: #2563eb;
        border: 1px solid #bfd5ff;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .stance-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 800;
        min-width: 74px;
        text-align: center;
    }

    .stance-hawk {
        background: #fdecec;
        color: #c62828;
        border: 1px solid #f7c9c9;
    }

    .stance-dove {
        background: #eaf7ee;
        color: #1f8a4c;
        border: 1px solid #c8ead3;
    }

    .stance-neutral {
        background: #f1f3f5;
        color: #59636e;
        border: 1px solid #dfe3e7;
    }

    .stance-unknown {
        background: #f6f6f6;
        color: #8a8f98;
        border: 1px solid #e7e7e7;
    }

    .score-hawk {
        color: #d32f2f;
        font-weight: 800;
    }

    .score-dove {
        color: #218c4a;
        font-weight: 800;
    }

    .score-neutral {
        color: #374151;
        font-weight: 700;
    }

    .fed-note {
        color: #6b7280;
        font-size: 0.81rem;
        line-height: 1.6;
        margin-top: 10px;
    }

    .score-card {
        margin-top: 20px;
        border: 1px solid #dfe5ec;
        border-radius: 10px;
        padding: 16px 18px 14px 18px;
        background: #fbfcfd;
    }

    .score-card-title {
        font-size: 0.98rem;
        font-weight: 750;
        color: #1f2937;
        margin-bottom: 12px;
    }

    .score-gradient {
        width: 100%;
        height: 16px;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                #169447 0%,
                #54b879 18%,
                #bfe3ca 40%,
                #f4f4f4 50%,
                #f7c5c5 62%,
                #ef7777 80%,
                #d92525 100%
            );
        border: 1px solid #e1e5e9;
    }

    .score-axis {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        margin-top: 6px;
        font-size: 0.76rem;
        color: #6b7280;
    }

    .score-axis span:nth-child(1) { text-align: left; }
    .score-axis span:nth-child(2),
    .score-axis span:nth-child(3),
    .score-axis span:nth-child(4) { text-align: center; }
    .score-axis span:nth-child(5) { text-align: right; }

    .score-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 0.79rem;
        font-weight: 750;
    }

    .score-label-dove { color: #218c4a; }
    .score-label-neutral { color: #59636e; }
    .score-label-hawk { color: #c62828; }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    /* Speaker Intelligence dashboard */
    .intel-card {
        background: #ffffff;
        border: 1px solid #e8edf2;
        border-radius: 12px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .intel-eyebrow {
        color: #8a97a6;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .intel-name {
        color: #1f2937;
        font-size: 1.28rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .intel-role {
        color: #8a97a6;
        font-size: 0.82rem;
        margin-bottom: 15px;
    }
    .intel-kpi {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid #f0f3f6;
    }
    .intel-kpi:last-child { border-bottom: none; }
    .intel-kpi-label { color: #7b8794; font-size: 0.82rem; }
    .intel-kpi-value { color: #263746; font-size: 1.05rem; font-weight: 800; }
    .intel-kpi-value.accent { color: #0c9aa6; }
    .intel-section-title {
        color: #263746;
        font-size: 0.98rem;
        font-weight: 800;
        margin: 0 0 10px 0;
    }
    .intel-news {
        border-left: 3px solid #0aa3ad;
        padding: 2px 0 2px 12px;
        margin: 0 0 15px 0;
    }
    .intel-news-meta { color: #8a97a6; font-size: 0.74rem; margin-bottom: 4px; }
    .intel-news-title { color: #263746; font-size: 0.88rem; font-weight: 700; line-height: 1.35; }
    .intel-news-score { color: #0c9aa6; font-size: 0.76rem; font-weight: 800; margin-top: 4px; }
    .intel-footnote { color: #8a97a6; font-size: 0.76rem; line-height: 1.5; margin-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MEMBER MASTER
# ============================================================

MEMBER_INFO = {
    "Kevin Warsh": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Philip Jefferson": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Michelle Bowman": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Michael Barr": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Lisa Cook": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Jerome Powell": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Christopher Waller": {
        "role": "Board of Governors",
        "voter": 1,
    },
    "Susan Collins": {
        "role": "Boston Fed",
        "voter": 0,
    },
    "John Williams": {
        "role": "New York Fed",
        "voter": 1,
    },
    "Anna Paulson": {
        "role": "Philadelphia Fed",
        "voter": 1,
    },
    "Beth Hammack": {
        "role": "Cleveland Fed",
        "voter": 1,
    },
    "Thomas Barkin": {
        "role": "Richmond Fed",
        "voter": 0,
    },
    "Cheryl Venable": {
        "role": "Atlanta Fed",
        "voter": 0,
    },
    "Austan Goolsbee": {
        "role": "Chicago Fed",
        "voter": 0,
    },
    "Alberto Musalem": {
        "role": "St. Louis Fed",
        "voter": 0,
    },
    "Neel Kashkari": {
        "role": "Minneapolis Fed",
        "voter": 1,
    },
    "Jeffrey Schmid": {
        "role": "Kansas City Fed",
        "voter": 0,
    },
    "Lorie Logan": {
        "role": "Dallas Fed",
        "voter": 1,
    },
    "Mary Daly": {
        "role": "San Francisco Fed",
        "voter": 0,
    },
}


# ============================================================
# IO
# ============================================================

@st.cache_data(show_spinner=False)
def load_json(filename: str):
    path = RESULTS_DIR / filename

    if not path.exists():
        return []

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []


def to_df(rows) -> pd.DataFrame:
    if not isinstance(rows, list):
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    return pd.json_normalize(
        rows
    )


def first_existing_column(
    df: pd.DataFrame,
    names,
):
    for name in names:
        if name in df.columns:
            return name

    return None


def parse_datetime_column(
    df: pd.DataFrame,
    column: str | None,
) -> pd.DataFrame:
    if (
        df.empty
        or not column
        or column not in df.columns
    ):
        return df

    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=True,
    )

    return df


# ============================================================
# LOAD RESULTS
# ============================================================

documents = to_df(
    load_json("documents.json")
)

segments = to_df(
    load_json("segments.json")
)

segment_stance = to_df(
    load_json("segment_stance.json")
)

member_stance = to_df(
    load_json("member_stance.json")
)

final_results = to_df(
    load_json("final_results.json")
)

coverage = to_df(
    load_json("coverage.json")
)

reuters_stance = to_df(
    load_json("reuters_stance.json")
)

news_documents = to_df(
    load_json("news_documents.json")
)

news_segments = to_df(
    load_json("news_segments.json")
)

news_segment_stance = to_df(
    load_json("news_segment_stance.json")
)

news_member_stance = to_df(
    load_json("news_member_stance.json")
)

# news_history는 pipeline 실행 중 계속 갱신되는 누적 파일이므로
# Streamlit cache를 거치지 않고 현재 파일 내용을 직접 읽는다.
_news_history_path = RESULTS_DIR / "news_history.json"

try:
    _news_history_raw = json.loads(
        _news_history_path.read_text(encoding="utf-8")
    )
except (
    OSError,
    json.JSONDecodeError,
):
    _news_history_raw = []

news_history = to_df(_news_history_raw)


# ============================================================
# NORMALIZE
# ============================================================

document_date_col = first_existing_column(
    documents,
    [
        "published_at",
        "date",
        "published",
    ],
)

documents = parse_datetime_column(
    documents,
    document_date_col,
)

news_date_col = first_existing_column(
    news_documents,
    [
        "published_at",
        "date",
        "published",
    ],
)

news_documents = parse_datetime_column(
    news_documents,
    news_date_col,
)

news_history_date_col = first_existing_column(
    news_history,
    [
        "published_at",
        "date",
    ],
)

news_history = parse_datetime_column(
    news_history,
    news_history_date_col,
)

news_member_name_col = first_existing_column(
    news_member_stance,
    [
        "member",
        "speaker",
        "name",
    ],
)

news_member_score_col = first_existing_column(
    news_member_stance,
    [
        "score",
        "stance_score",
        "final_score",
        "weighted_score",
    ],
)

news_member_label_col = first_existing_column(
    news_member_stance,
    [
        "stance",
        "label",
        "final_stance",
    ],
)

member_name_col = first_existing_column(
    member_stance,
    [
        "member",
        "speaker",
        "name",
    ],
)

member_score_col = first_existing_column(
    member_stance,
    [
        "score",
        "stance_score",
        "final_score",
        "weighted_score",
    ],
)

member_label_col = first_existing_column(
    member_stance,
    [
        "stance",
        "label",
        "final_stance",
    ],
)


# ============================================================
# HELPERS
# ============================================================

def member_meta(name):
    return MEMBER_INFO.get(
        str(name),
        {
            "role": "",
            "voter": 0,
        },
    )


def source_group(value):
    value = str(
        value
        or ""
    ).lower()

    if value.startswith("regional_"):
        return "Regional Official"

    if value in {
        "fed_speech",
        "fed_testimony",
    }:
        return "Board Official"

    if value == "google_news":
        return "News"

    if value.startswith("news_rss_"):
        return "News"

    return value or "Unknown"


def label_from_score(value):
    try:
        score = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return "UNKNOWN"

    if score >= 0.20:
        return "HAWKISH"

    if score <= -0.20:
        return "DOVISH"

    return "NEUTRAL"


def normalize_label(value, score=None):
    label = str(
        value
        or ""
    ).upper().strip()

    if "HAWK" in label:
        return "HAWKISH"

    if "DOVE" in label:
        return "DOVISH"

    if "NEUT" in label:
        return "NEUTRAL"

    if score is not None and pd.notna(score):
        return label_from_score(score)

    return "UNKNOWN"


def stance_badge(label):
    label = normalize_label(label)

    css_class = {
        "HAWKISH": "stance-hawk",
        "DOVISH": "stance-dove",
        "NEUTRAL": "stance-neutral",
        "UNKNOWN": "stance-unknown",
    }.get(
        label,
        "stance-unknown",
    )

    return (
        f'<span class="stance-badge {css_class}">'
        f'{html.escape(label)}'
        f'</span>'
    )

def score_html(value):
    if value is None or pd.isna(value):
        return "-"

    score = max(
        -1.0,
        min(1.0, float(value))
    )

    # Neutral = white
    neutral = (255, 255, 255)

    # Dovish = green
    dove = (105, 190, 130)

    # Hawkish = red
    hawk = (230, 105, 105)

    if score < 0:
        strength = abs(score)

        r = int(
            neutral[0]
            + (dove[0] - neutral[0]) * strength
        )
        g = int(
            neutral[1]
            + (dove[1] - neutral[1]) * strength
        )
        b = int(
            neutral[2]
            + (dove[2] - neutral[2]) * strength
        )

    else:
        strength = score

        r = int(
            neutral[0]
            + (hawk[0] - neutral[0]) * strength
        )
        g = int(
            neutral[1]
            + (hawk[1] - neutral[1]) * strength
        )
        b = int(
            neutral[2]
            + (hawk[2] - neutral[2]) * strength
        )

    background = f"rgb({r}, {g}, {b})"

    return (
        f'<span style="'
        f'display:block;'
        f'width:100%;'
        f'padding:5px 8px;'
        f'border-radius:5px;'
        f'background:{background};'
        f'color:#1f2937;'
        f'font-weight:700;'
        f'text-align:center;'
        f'box-sizing:border-box;'
        f'">'
        f'{score:+.2f}'
        f'</span>'
    )

def official_remark_link(date_value, url_value):
    date_text = str(date_value or "-")
    url_text = str(url_value or "").strip()

    if not url_text:
        return html.escape(date_text)

    return (
        f'<a href="{html.escape(url_text, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">'
        f'{html.escape(date_text)}</a>'
    )


def voter_html(is_voter):
    if is_voter:
        return (
            '<span class="voter-badge">'
            'Voter'
            '</span>'
        )

    return "—"


def get_reuters_stance_map():
    data = load_json("reuters_stance.json")

    if not isinstance(data, dict):
        return {}

    members = data.get("members", {})

    return {
        str(name).strip(): str(info.get("stance", "")).strip().upper()
        for name, info in members.items()
        if isinstance(info, dict) and info.get("stance")
    }


def reuters_stance_badge(value):
    label = str(value or "").strip().upper()
    css_class = {
        "DOVE": "stance-dove",
        "DOVISH": "stance-dove",
        "CENTRIST": "stance-neutral",
        "NEUTRAL": "stance-neutral",
        "HAWKISH": "stance-hawk",
        "HAWK": "stance-hawk",
    }.get(label, "stance-unknown")

    return (
        f'<span class="stance-badge {css_class}">'
        f'{html.escape(label or "—")}'
        f'</span>'
    )


def get_member_summary_df():
    rows = []
    reuters_map = get_reuters_stance_map()

    for member, info in MEMBER_INFO.items():
        row = {
            "Member": member,
            "Role": info["role"],
            "Voter": bool(info["voter"]),
            "Reuters Stance": reuters_map.get(member, ""),
            "Stance": "UNKNOWN",
            "Score": None,
            "Latest": None,
            "Latest URL": None,
            "Official": 0,
        }

        if (
            not member_stance.empty
            and member_name_col
        ):
            matched = member_stance[
                member_stance[
                    member_name_col
                ]
                == member
            ]

            if not matched.empty:
                record = matched.iloc[0]

                if member_score_col:
                    row["Score"] = pd.to_numeric(
                        record.get(
                            member_score_col
                        ),
                        errors="coerce",
                    )

                if member_label_col:
                    row["Stance"] = normalize_label(
                        record.get(
                            member_label_col
                        ),
                        row["Score"],
                    )

                else:
                    row["Stance"] = normalize_label(
                        "",
                        row["Score"],
                    )

        if (
            not documents.empty
            and "speaker" in documents.columns
        ):
            member_docs = documents[
                documents["speaker"]
                == member
            ].copy()

            if not member_docs.empty:
                if "source" in member_docs.columns:
                    groups = member_docs[
                        "source"
                    ].map(
                        source_group
                    )

                    row["Official"] = int(
                        groups.isin(
                            [
                                "Board Official",
                                "Regional Official",
                            ]
                        ).sum()
                    )

                if document_date_col:
                    official_docs = member_docs.copy()

                    if "source" in official_docs.columns:
                        official_groups = (
                            official_docs[
                                "source"
                            ].map(
                                source_group
                            )
                        )

                        official_docs = (
                            official_docs[
                                official_groups.isin(
                                    [
                                        "Board Official",
                                        "Regional Official",
                                    ]
                                )
                            ]
                        )

                    if not official_docs.empty:
                        dated_docs = official_docs.dropna(
                            subset=[document_date_col]
                        ).sort_values(
                            document_date_col,
                            ascending=False,
                        )

                        if not dated_docs.empty:
                            latest_record = dated_docs.iloc[0]
                            latest = latest_record.get(document_date_col)

                            if pd.notna(latest):
                                row["Latest"] = latest.strftime("%Y-%m-%d")

                            for url_col in ["url", "document_url", "link"]:
                                if url_col in dated_docs.columns:
                                    latest_url = latest_record.get(url_col)
                                    if pd.notna(latest_url) and str(latest_url).strip():
                                        row["Latest URL"] = str(latest_url).strip()
                                        break

        rows.append(row)

    return pd.DataFrame(rows)


def get_news_df():
    if news_documents.empty:
        return pd.DataFrame()

    return news_documents.copy()


def get_official_df():
    if (
        documents.empty
        or "source" not in documents.columns
    ):
        return pd.DataFrame()

    groups = documents[
        "source"
    ].map(
        source_group
    )

    return documents[
        groups.isin(
            [
                "Board Official",
                "Regional Official",
            ]
        )
    ].copy()



def get_news_member_summary_df(
    selected,
):
    rows = []

    for member in selected:
        meta = member_meta(
            member
        )

        member_docs = (
            news_documents[
                news_documents["speaker"]
                == member
            ].copy()
            if (
                not news_documents.empty
                and "speaker"
                in news_documents.columns
            )
            else pd.DataFrame()
        )

        score = None
        stance = "UNKNOWN"

        if (
            not news_member_stance.empty
            and news_member_name_col
        ):
            matched = (
                news_member_stance[
                    news_member_stance[
                        news_member_name_col
                    ]
                    == member
                ]
            )

            if not matched.empty:
                record = matched.iloc[0]

                if news_member_score_col:
                    score = pd.to_numeric(
                        record.get(
                            news_member_score_col
                        ),
                        errors="coerce",
                    )

                if news_member_label_col:
                    stance = normalize_label(
                        record.get(
                            news_member_label_col
                        ),
                        score,
                    )
                else:
                    stance = normalize_label(
                        "",
                        score,
                    )

        latest = None

        if (
            not member_docs.empty
            and news_date_col
        ):
            latest_value = member_docs[
                news_date_col
            ].max()

            if pd.notna(
                latest_value
            ):
                latest = (
                    latest_value.strftime(
                        "%Y-%m-%d"
                    )
                )

        rows.append(
            {
                "Member": member,
                "Role": meta["role"],
                "Voter": bool(
                    meta["voter"]
                ),
                "News Events": len(
                    member_docs
                ),
                "Stance": stance,
                "Score": score,
                "Latest News": latest,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["Score"] = pd.to_numeric(
        result["Score"],
        errors="coerce",
    )

    return result


def build_news_event_scores():
    if (
        news_documents.empty
        or news_segments.empty
        or news_segment_stance.empty
    ):
        return pd.DataFrame()

    stance_id = first_existing_column(
        news_segment_stance,
        ["segment_id"],
    )

    segment_id = first_existing_column(
        news_segments,
        ["segment_id"],
    )

    if not stance_id or not segment_id:
        return pd.DataFrame()

    keep_cols = [
        column
        for column in [
            segment_id,
            "document_id",
            "speaker",
        ]
        if column
        in news_segments.columns
    ]

    merged = (
        news_segment_stance.merge(
            news_segments[
                keep_cols
            ],
            left_on=stance_id,
            right_on=segment_id,
            how="left",
        )
    )

    if "policy_relevant" in merged.columns:
        merged = merged[
            merged[
                "policy_relevant"
            ]
            == True
        ]

    if (
        merged.empty
        or "document_id"
        not in merged.columns
        or "score"
        not in merged.columns
    ):
        return pd.DataFrame()

    event_scores = (
        merged.groupby(
            [
                "document_id",
                "speaker",
            ],
            as_index=False,
        )["score"]
        .mean()
    )

    document_id_col = first_existing_column(
        news_documents,
        ["document_id", "id"],
    )

    if not document_id_col:
        return event_scores

    meta_cols = [
        column
        for column in [
            document_id_col,
            "title",
            "url",
            news_date_col,
        ]
        if column
        and column
        in news_documents.columns
    ]

    return event_scores.merge(
        news_documents[
            meta_cols
        ],
        left_on="document_id",
        right_on=document_id_col,
        how="left",
    )


def get_recent_news_history(
    selected,
    days=NEWS_RECENT_DAYS,
):
    """
    최근 N일 news history.
    Recent News Summary와 current news score 계산에 사용한다.
    """

    if (
        news_history.empty
        or not news_history_date_col
    ):
        return pd.DataFrame()

    result = news_history.copy()

    if "speaker" in result.columns:
        result = result[
            result["speaker"].isin(
                selected
            )
        ]

    cutoff = (
        pd.Timestamp.now(
            tz="UTC"
        )
        - pd.Timedelta(
            days=days
        )
    )

    return result[
        result[
            news_history_date_col
        ]
        >= cutoff
    ].copy()


def get_ytd_news_history(
    selected,
):
    """
    2026-01-01부터 현재까지의 news history.
    Speaker Trend 전용이다.
    """

    if (
        news_history.empty
        or not news_history_date_col
    ):
        return pd.DataFrame()

    result = news_history.copy()

    if "speaker" in result.columns:
        result = result[
            result["speaker"].isin(
                selected
            )
        ]

    start_date = pd.Timestamp(
        NEWS_TREND_START,
        tz="UTC",
    )

    return result[
        result[
            news_history_date_col
        ]
        >= start_date
    ].copy()


def get_news_current_score_df(
    selected,
    days=NEWS_CURRENT_SCORE_DAYS,
):
    """
    Combined Score에 사용할 현재 News Score.

    2026년 전체 평균이 아니라
    최근 N일 policy-relevant 뉴스 score 평균을 사용한다.
    """

    history = get_recent_news_history(
        selected,
        days=days,
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "Member",
                "Score",
            ]
        )

    if "policy_relevant" in history.columns:
        history = history[
            history[
                "policy_relevant"
            ]
            == True
        ]

    if (
        history.empty
        or "speaker"
        not in history.columns
        or "score"
        not in history.columns
    ):
        return pd.DataFrame(
            columns=[
                "Member",
                "Score",
            ]
        )

    history = history.copy()

    history["score"] = pd.to_numeric(
        history["score"],
        errors="coerce",
    )

    history = history.dropna(
        subset=[
            "score"
        ]
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "Member",
                "Score",
            ]
        )

    return (
        history.groupby(
            "speaker",
            as_index=False,
        )["score"]
        .mean()
        .rename(
            columns={
                "speaker":
                    "Member",
                "score":
                    "Score",
            }
        )
    )


def get_combined_score_df(
    selected,
):
    """
    Official Score + 최근 30일 News Score.

    둘 다 있으면 50:50 단순 평균,
    하나만 있으면 존재하는 score를 사용한다.
    """

    official = get_member_summary_df()

    official = official[
        official[
            "Member"
        ].isin(
            selected
        )
    ].copy()

    news_summary = (
        get_news_current_score_df(
            selected,
            days=NEWS_CURRENT_SCORE_DAYS,
        )
    )

    news_map = {
        row["Member"]:
            row["Score"]
        for _, row
        in news_summary.iterrows()
    }

    rows = []

    for _, row in official.iterrows():
        member = row[
            "Member"
        ]

        official_score = pd.to_numeric(
            row.get(
                "Score"
            ),
            errors="coerce",
        )

        news_score = pd.to_numeric(
            news_map.get(
                member
            ),
            errors="coerce",
        )

        if (
            pd.notna(
                official_score
            )
            and pd.notna(
                news_score
            )
        ):
            combined = (
                COMBINED_OFFICIAL_WEIGHT
                * float(
                    official_score
                )
                +
                COMBINED_NEWS_WEIGHT
                * float(
                    news_score
                )
            )

        elif pd.notna(
            official_score
        ):
            combined = float(
                official_score
            )

        elif pd.notna(
            news_score
        ):
            combined = float(
                news_score
            )

        else:
            combined = None

        rows.append(
            {
                "Member":
                    member,
                "Role":
                    row["Role"],
                "Voter":
                    row["Voter"],
                "Official Score":
                    official_score,
                "News 30D Score":
                    news_score,
                "Combined Score":
                    combined,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result = result.sort_values(
            [
                "Combined Score",
                "Member",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )

    return result



def build_official_event_scores():
    """Official document별 policy-relevant segment score 평균."""
    if documents.empty or segments.empty or segment_stance.empty:
        return pd.DataFrame()

    stance_id = first_existing_column(segment_stance, ["segment_id"])
    segment_id = first_existing_column(segments, ["segment_id"])
    if not stance_id or not segment_id:
        return pd.DataFrame()

    keep_cols = [
        c for c in [segment_id, "document_url", "speaker"]
        if c in segments.columns
    ]

    merged = segment_stance.merge(
        segments[keep_cols],
        left_on=stance_id,
        right_on=segment_id,
        how="left",
    )

    if "policy_relevant" in merged.columns:
        merged = merged[merged["policy_relevant"] == True]

    if (
        merged.empty
        or "document_url" not in merged.columns
        or "score" not in merged.columns
    ):
        return pd.DataFrame()

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged = merged.dropna(subset=["score"])

    event_scores = (
        merged.groupby(["document_url", "speaker"], as_index=False)["score"]
        .mean()
    )

    if "url" not in documents.columns:
        return event_scores

    meta_cols = [
        c for c in ["url", "title", document_date_col]
        if c and c in documents.columns
    ]

    return event_scores.merge(
        documents[meta_cols],
        left_on="document_url",
        right_on="url",
        how="left",
    )

def get_monthly_member_scores(member):
    """2026 월별 Official / News / Combined 평균."""
    end_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    months = pd.period_range(NEWS_TREND_START, end_month, freq="M")
    result = pd.DataFrame({"Month": [str(m) for m in months]})

    official = build_official_event_scores()
    if (
        not official.empty
        and "speaker" in official.columns
        and document_date_col
        and document_date_col in official.columns
    ):
        official = official[official["speaker"] == member].copy()
        official[document_date_col] = pd.to_datetime(
            official[document_date_col], errors="coerce", utc=True
        )
        official = official.dropna(subset=[document_date_col, "score"])
        if not official.empty:
            official["Month"] = official[document_date_col].dt.strftime("%Y-%m")
            monthly_official = (
                official.groupby("Month", as_index=False)["score"]
                .mean()
                .rename(columns={"score": "Official"})
            )
            result = result.merge(monthly_official, on="Month", how="left")

    if "Official" not in result.columns:
        result["Official"] = pd.NA

    history = get_ytd_news_history([member])
    if (
        not history.empty
        and news_history_date_col
        and "score" in history.columns
    ):
        if "policy_relevant" in history.columns:
            history = history[history["policy_relevant"] == True].copy()
        history["score"] = pd.to_numeric(history["score"], errors="coerce")
        history = history.dropna(subset=[news_history_date_col, "score"])
        if not history.empty:
            history["Month"] = history[news_history_date_col].dt.strftime("%Y-%m")
            monthly_news = (
                history.groupby("Month", as_index=False)["score"]
                .mean()
                .rename(columns={"score": "News"})
            )
            result = result.merge(monthly_news, on="Month", how="left")

    if "News" not in result.columns:
        result["News"] = pd.NA

    result["Official"] = pd.to_numeric(result["Official"], errors="coerce")
    result["News"] = pd.to_numeric(result["News"], errors="coerce")

    def combine(row):
        o, n = row["Official"], row["News"]
        if pd.notna(o) and pd.notna(n):
            return COMBINED_OFFICIAL_WEIGHT * o + COMBINED_NEWS_WEIGHT * n
        if pd.notna(o):
            return o
        if pd.notna(n):
            return n
        return None

    result["Combined"] = result.apply(combine, axis=1)
    return result



def render_member_table(
    summary: pd.DataFrame,
):
    rows_html = []

    for _, row in summary.iterrows():
        row_class = (
            "voter-row"
            if bool(
                row["Voter"]
            )
            else ""
        )

        row_html = f"""
        <tr class="{row_class}">
            <td>{html.escape(str(row["Member"]))}</td>
            <td>{html.escape(str(row["Role"]))}</td>
            <td>{voter_html(bool(row["Voter"]))}</td>
            <td>{reuters_stance_badge(row.get("Reuters Stance", ""))}</td>
            <td>{score_html(row["Score"])}</td>
            <td>{official_remark_link(row.get("Latest"), row.get("Latest URL"))}</td>
        </tr>
        """

        rows_html.append(
            textwrap.dedent(
                row_html
            ).strip()
        )

    table_html = f"""
    <div class="fed-table-wrap">
        <table class="fed-table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Role</th>
                    <th>Voter</th>
                    <th>Reuters Stance</th>
                    <th>Score (-1 to +1)</th>
                    <th>Last Official Remark</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """

    st.markdown(
        textwrap.dedent(
            table_html
        ).strip(),
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Filter"
)

all_members = list(
    MEMBER_INFO.keys()
)

selected_members = st.sidebar.multiselect(
    "Fed Member",
    options=all_members,
    default=all_members,
)

voters_only = st.sidebar.checkbox(
    "투표권자만 보기",
    value=False,
)

if voters_only:
    selected_members = [
        member
        for member in selected_members
        if MEMBER_INFO[
            member
        ]["voter"]
    ]

if st.sidebar.button(
    "데이터 새로고침"
):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# 3 TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "1. Fed Member Dashboard",
        "2. Speaker Intelligence",
        "3. Methodology",
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:
    summary = get_member_summary_df()

    summary = summary[
        summary["Member"].isin(
            selected_members
        )
    ].copy()

    summary["Score"] = pd.to_numeric(
        summary["Score"],
        errors="coerce",
    )

    summary = summary.sort_values(
        by=[
            "Score",
            "Member",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    official_total = int(
        summary[
            "Official"
        ].sum()
    )

    scored_members = int(
        summary[
            "Score"
        ].notna().sum()
    )

    st.markdown(
        f"""
        <div class="fed-topline">
            <div>
                <h1 class="fed-title" style="font-size:22px;">
                    Current Fed Member Stance
                </h1>
            </div>
            <div class="fed-mini-metrics">
                Members&nbsp;&nbsp;<b>{len(summary)}</b>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Official Documents&nbsp;&nbsp;<b>{official_total}</b>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Scored Members&nbsp;&nbsp;<b>{scored_members}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fed-info-box">
            <span class="fed-info-icon">i</span>
            <b>Reuters Stance</b>는 Reuters의 Doves and Hawks 분류를 외부 benchmark로 표시합니다.
            <b>Score</b>는 연준 및 각 지역 연은 공식 홈페이지 발언을 LLM으로 분석한 값으로,
            -1에 가까울수록 Dovish, 0은 Neutral, +1에 가까울수록 Hawkish입니다.
            Reuters 분류는 LLM score 및 Combined Score 계산에는 사용하지 않습니다.
            <b>Last Official Remark</b> 날짜를 클릭하면 해당 공식 원문으로 이동합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_member_table(
        summary
    )

    st.markdown(
        """
        <div class="fed-note">
            * Score: -1에 가까울수록 Dovish, 0은 Neutral, +1에 가까울수록 Hawkish입니다.<br>
            ** Score가 '-'인 경우: 공식 발언이 아직 수집되지 않았거나,
            수집된 발언 중 policy-relevant segment가 충분하지 않아
            최종 score가 생성되지 않은 경우입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="score-card">
            <div class="score-card-title">Score Distribution (Policy Tone)</div>
            <div class="score-gradient"></div>
            <div class="score-axis">
                <span>-1.0</span>
                <span>-0.5</span>
                <span>0</span>
                <span>+0.5</span>
                <span>+1.0</span>
            </div>
            <div class="score-labels">
                <span class="score-label-dove">Dovish</span>
                <span class="score-label-neutral">Neutral</span>
                <span class="score-label-hawk">Hawkish</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown(
        """
        <div class="fed-topline">
            <div><h1 class="fed-title">Speaker Intelligence Dashboard</h1></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "최근 14일 뉴스 발언과 2026 월별 Official·News stance, "
        "현재 Combined Score를 한 화면에서 확인합니다."
    )

    available_members = [m for m in selected_members if m in MEMBER_INFO]

    if not available_members:
        st.info("선택된 Fed member가 없습니다.")
    else:
        trend_member = st.selectbox(
            "Fed Member",
            options=available_members,
            key="speaker_dashboard_member",
        )

        official_summary = get_member_summary_df()
        official_row = official_summary[official_summary["Member"] == trend_member]

        official_score = None
        official_stance = "UNKNOWN"
        if not official_row.empty:
            value = pd.to_numeric(official_row.iloc[0].get("Score"), errors="coerce")
            if pd.notna(value):
                official_score = float(value)
                official_stance = normalize_label(
                    official_row.iloc[0].get("Stance"), official_score
                )

        current_news = get_news_current_score_df(
            [trend_member], days=NEWS_CURRENT_SCORE_DAYS
        )
        news_score = None
        if not current_news.empty:
            value = pd.to_numeric(current_news.iloc[0].get("Score"), errors="coerce")
            if pd.notna(value):
                news_score = float(value)

        combined_rows = get_combined_score_df([trend_member])
        combined_score = None
        if not combined_rows.empty:
            value = pd.to_numeric(
                combined_rows.iloc[0].get("Combined Score"), errors="coerce"
            )
            if pd.notna(value):
                combined_score = float(value)

        recent_news = get_recent_news_history([trend_member], days=NEWS_RECENT_DAYS)
        monthly = get_monthly_member_scores(trend_member)

        role = member_meta(trend_member).get("role", "")
        combined_stance = normalize_label("", combined_score)
        fmt = lambda v: "-" if v is None or pd.isna(v) else f"{float(v):+.2f}"

        # Reference-style 3-column dashboard: member card / trend / recent news
        profile_col, trend_col, news_col = st.columns([0.78, 1.55, 1.15], gap="large")

        with profile_col:
            st.markdown(
                f"""
                <div class="intel-card">
                    <div class="intel-eyebrow">Fed Member</div>
                    <div class="intel-name">{html.escape(trend_member)}</div>
                    <div class="intel-role">{html.escape(role)}</div>
                    <div class="intel-kpi"><span class="intel-kpi-label">Official</span><span class="intel-kpi-value">{fmt(official_score)}</span></div>
                    <div class="intel-kpi"><span class="intel-kpi-label">News 30D</span><span class="intel-kpi-value">{fmt(news_score)}</span></div>
                    <div class="intel-kpi"><span class="intel-kpi-label">Combined</span><span class="intel-kpi-value accent">{fmt(combined_score)}</span></div>
                    <div class="intel-kpi"><span class="intel-kpi-label">Articles · 14D</span><span class="intel-kpi-value">{len(recent_news)}</span></div>
                    <div style="margin-top:16px">
                        <div class="intel-eyebrow">Current Stance</div>
                        {stance_badge(combined_stance)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with trend_col:
            st.markdown('<div class="intel-section-title">Monthly Stance · 2026</div>', unsafe_allow_html=True)
            chart_data = monthly.melt(
                id_vars=["Month"],
                value_vars=["Official", "News", "Combined"],
                var_name="Series",
                value_name="Score",
            )
            chart_data["Score"] = pd.to_numeric(chart_data["Score"], errors="coerce")
            chart_data["MonthDate"] = pd.to_datetime(
                chart_data["Month"] + "-01", errors="coerce"
            )
            chart_data = chart_data.dropna(subset=["MonthDate"])

            if chart_data["Score"].notna().sum() == 0:
                st.info("월별 stance score가 없습니다.")
            else:
                month_start = pd.Timestamp(NEWS_TREND_START)
                month_end = pd.Timestamp.now().normalize().replace(day=1)
                line = (
                    alt.Chart(chart_data)
                    .mark_line(point=True, strokeWidth=2.4)
                    .encode(
                        x=alt.X(
                            "MonthDate:T",
                            title=None,
                            scale=alt.Scale(domain=[month_start, month_end]),
                            axis=alt.Axis(format="%b", grid=False, labelColor="#7b8794", title=None),
                        ),
                        y=alt.Y(
                            "Score:Q",
                            title=None,
                            scale=alt.Scale(domain=[-1, 1]),
                            axis=alt.Axis(grid=True, gridColor="#edf1f4", labelColor="#7b8794"),
                        ),
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(
                                domain=["Official", "News", "Combined"],
                                range=["#0aa3ad", "#f3aa3d", "#4f7df3"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=[
                            alt.Tooltip("Month:N", title="Month"),
                            alt.Tooltip("Series:N", title="Source"),
                            alt.Tooltip("Score:Q", title="Score", format="+.2f"),
                        ],
                    )
                    .properties(height=315)
                )
                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                    color="#cfd8df", strokeDash=[4, 4]
                ).encode(y="y:Q")
                st.altair_chart(line + zero, use_container_width=True)

            st.markdown(
                '<div class="intel-footnote">빈 월은 0점으로 채우지 않습니다. 해당 월에 실제 scoring 데이터가 없다는 뜻입니다.</div>',
                unsafe_allow_html=True,
            )

        with news_col:
            st.markdown('<div class="intel-section-title">Recent News Remarks · 14D</div>', unsafe_allow_html=True)
            if recent_news.empty:
                st.info("최근 14일 뉴스가 없습니다.")
            else:
                if news_history_date_col:
                    recent_news = recent_news.sort_values(news_history_date_col, ascending=False)

                for _, row in recent_news.head(6).iterrows():
                    published = row.get(news_history_date_col, "")
                    if pd.notna(published) and hasattr(published, "strftime"):
                        published = published.strftime("%b %d")
                    source_value = str(row.get("source", ""))
                    publisher = source_value.split("|", 1)[1] if "|" in source_value else source_value
                    title = str(row.get("title", ""))
                    score = pd.to_numeric(row.get("score"), errors="coerce")
                    stance = normalize_label(row.get("stance", ""), score)
                    score_text = "-" if pd.isna(score) else f"{float(score):+.2f}"
                    st.markdown(
                        f"""
                        <div class="intel-news">
                            <div class="intel-news-meta">{html.escape(str(published))} · {html.escape(publisher)}</div>
                            <div class="intel-news-title">{html.escape(title)}</div>
                            <div class="intel-news-score">{html.escape(stance)} · {score_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("View news details"):
                    for _, row in recent_news.head(20).iterrows():
                        published = row.get(news_history_date_col, "")
                        if pd.notna(published) and hasattr(published, "strftime"):
                            published = published.strftime("%Y-%m-%d")
                        title = str(row.get("title", ""))
                        st.markdown(f"**{published}** · {title}")
                        summary_text = str(row.get("summary", ""))
                        if summary_text:
                            st.caption(summary_text[:700])
                        evidence = str(row.get("evidence", ""))
                        reasoning = str(row.get("reasoning", ""))
                        if evidence:
                            st.markdown(f"**Evidence**  \n{evidence}")
                        if reasoning:
                            st.markdown(f"**Reasoning**  \n{reasoning}")
                        url = row.get("url", "")
                        if url:
                            st.markdown(f"[기사 열기]({url})")
                        st.divider()

render_policy_signal_market_reaction(
    trend_member
)

# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown(
        """
        <div class="fed-topline">
            <div>
                <h1 class="fed-title">Methodology</h1>
                <div style="color:#7b8794;font-size:0.92rem;margin-top:4px;">
                    How Fed Speaker Monitor V2 collects, filters and scores Fed communication
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    official_count = len(get_official_df())
    news_count = len(news_history) if not news_history.empty else len(news_documents)

    relevant_count = 0
    if not segment_stance.empty and "policy_relevant" in segment_stance.columns:
        relevant_count += int((segment_stance["policy_relevant"] == True).sum())
    if not news_history.empty and "policy_relevant" in news_history.columns:
        relevant_count += int((news_history["policy_relevant"] == True).sum())

    scored_count = 0
    if not segment_stance.empty and "score" in segment_stance.columns:
        scored_count += int(pd.to_numeric(segment_stance["score"], errors="coerce").notna().sum())
    if not news_history.empty and "score" in news_history.columns:
        scored_count += int(pd.to_numeric(news_history["score"], errors="coerce").notna().sum())

    st.markdown(
        """
        <style>
        .method-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: #ffffff;
            padding: 18px 18px 16px 18px;
            min-height: 315px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.03);
        }
        .method-card-title {
            color: #1f2937;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 13px;
        }
        .method-step {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.55;
        }
        .method-arrow {
            color: #a0a9b4;
            text-align: center;
            font-size: 0.78rem;
            line-height: 1.15;
            margin: 2px 0;
        }
        .method-pill {
            display: inline-block;
            border-radius: 999px;
            background: #f3f6f9;
            border: 1px solid #e3e8ee;
            color: #536171;
            font-size: 0.72rem;
            font-weight: 750;
            padding: 3px 8px;
            margin: 2px 2px 2px 0;
        }
        .method-metric {
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding: 10px 0;
            border-bottom:1px solid #eef1f4;
            color:#667382;
            font-size:0.82rem;
        }
        .method-metric:last-child { border-bottom:none; }
        .method-metric b { color:#263746; font-size:1.03rem; }
        .reference-board {
            margin-top: 18px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: #ffffff;
            padding: 20px 22px 18px 22px;
        }
        .reference-title {
            color:#1f2937;
            font-size:1.05rem;
            font-weight:800;
            margin-bottom:4px;
        }
        .reference-subtitle {
            color:#8a97a6;
            font-size:0.80rem;
            margin-bottom:14px;
        }
        .reference-section {
            color:#8a97a6;
            font-size:0.70rem;
            font-weight:800;
            letter-spacing:0.08em;
            margin:12px 0 6px 0;
        }
        .reference-row {
            display:grid;
            grid-template-columns: minmax(190px, 0.8fr) minmax(260px, 1.6fr);
            gap:18px;
            padding:8px 0;
            border-bottom:1px solid #f0f2f4;
            font-size:0.82rem;
            line-height:1.45;
        }
        .reference-row:last-child { border-bottom:none; }
        .reference-row a {
            color:#2563eb;
            text-decoration:none;
            font-weight:750;
        }
        .reference-desc { color:#667382; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.05, 0.90, 1.05, 0.82], gap="medium")

    with c1:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-card-title">Fed Speaker Monitor V2 Pipeline</div>
                <div class="method-step"><b>Official</b> · Fed Board RSS + Regional Fed pages</div>
                <div class="method-arrow">↓</div>
                <div class="method-step">Incremental collection → junk / URL / title dedup</div>
                <div class="method-arrow">↓</div>
                <div class="method-step">Segmentation → LLM policy relevance → stance score</div>
                <div class="method-arrow">↓</div>
                <div class="method-step"><b>News</b> · Finlight primary + Google fallback</div>
                <div class="method-arrow">↓</div>
                <div class="method-step">Event clustering → attribution / relevance → history</div>
                <div class="method-arrow">↓</div>
                <div class="method-step"><b>Official + recent 30D News → Combined Score</b></div>
                <div style="margin-top:12px">
                    <span class="method-pill">-1 Dovish</span>
                    <span class="method-pill">0 Neutral</span>
                    <span class="method-pill">+1 Hawkish</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-card-title">Content Type</div>
                <div class="method-step"><b>PRESCRIPTIVE</b><br>향후 정책 방향이나 바람직한 조치를 직접 시사</div>
                <div style="height:10px"></div>
                <div class="method-step"><b>DESCRIPTIVE</b><br>경제·물가·고용 상황을 설명</div>
                <div style="height:10px"></div>
                <div class="method-step"><b>MIXED</b><br>설명과 정책적 시사가 함께 존재</div>
                <div style="height:10px"></div>
                <div class="method-step"><b>IRRELEVANT</b><br>통화정책 stance 판단과 관련성이 낮음</div>
                <div style="margin-top:14px" class="method-step">
                    <b>policy_relevant=False</b>는 최종 stance aggregation에서 제외
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="method-card">
                <div class="method-card-title">News Event Dedup</div>
                <div class="method-step">동일 Fed 발언의 반복·재배포 기사를 하나의 event로 묶습니다.</div>
                <div style="height:10px"></div>
                <div class="method-step">① Same speaker</div>
                <div class="method-step">② Nearby publication date</div>
                <div class="method-step">③ title + article lead similarity</div>
                <div class="method-step">④ Semantic event cluster</div>
                <div class="method-step">⑤ Representative article</div>
                <div class="method-step">⑥ Relevance filter</div>
                <div style="margin-top:14px">
                    <span class="method-pill">±4 days</span>
                    <span class="method-pill">similarity ≥ 0.78</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="method-card">
                <div class="method-card-title">Data Coverage</div>
                <div class="method-metric"><span>Fed Members</span><b>{len(MEMBER_INFO):,}</b></div>
                <div class="method-metric"><span>Official Docs</span><b>{official_count:,}</b></div>
                <div class="method-metric"><span>News History</span><b>{news_count:,}</b></div>
                <div class="method-metric"><span>Policy Relevant</span><b>{relevant_count:,}</b></div>
                <div class="method-metric"><span>Scored Items</span><b>{scored_count:,}</b></div>
                <div style="margin-top:12px" class="method-step">Counts reflect the currently loaded result files.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    reference_html = """<div class="reference-board">
<div class="reference-title">References &amp; Open-source Inspiration</div>
<div class="reference-subtitle">Research and public projects used as methodological references. Links open the original source.</div>
<div class="reference-section">RESEARCH</div>
<div class="reference-row"><div><a href="https://www.bis.org/publ/work1215.htm" target="_blank">BIS · CB-LMs</a></div><div class="reference-desc">Central-bank-domain language models; reference for domain-specific monetary-policy text classification.</div></div>
<div class="reference-row"><div><a href="https://www.bis.org/publ/work1253.htm" target="_blank">BIS · Word2Prices</a></div><div class="reference-desc">Reference for extracting structured economic information and signals from text.</div></div>
<div class="reference-row"><div><a href="https://www.bis.org/" target="_blank">BIS Research</a></div><div class="reference-desc">Additional central-bank communication and monetary-policy NLP research used for methodology review.</div></div>
<div class="reference-section">OPEN SOURCE</div>
<div class="reference-row"><div><a href="https://github.com/usydnlp/FedNLP" target="_blank">FedNLP</a></div><div class="reference-desc">Federal Reserve communication NLP datasets and modeling reference.</div></div>
<div class="reference-row"><div><a href="https://github.com/gtfintechlab/FOMC-NLP" target="_blank">FOMC-NLP</a></div><div class="reference-desc">FOMC text analysis and hawkish / dovish classification reference.</div></div>
<div class="reference-section">PROJECT DESIGN</div>
<div class="reference-row"><div>Official + News separation</div><div class="reference-desc">Official remarks remain the primary stance source; news is retained as a separate event layer and recent signal.</div></div>
<div class="reference-row"><div>Reuters Stance</div><div class="reference-desc">Displayed as an external benchmark only. It is not used in LLM score or Combined Score calculation.</div></div>
</div>"""
    st.markdown(reference_html, unsafe_allow_html=True)

