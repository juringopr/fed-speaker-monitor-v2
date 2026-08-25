from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

from fed_speaker_monitor_v2.models import Document


FINLIGHT_API_URL = "https://api.finlight.me/v2/articles"


FED_MEMBERS = [
    "Kevin Warsh",
    "Philip Jefferson",
    "Michelle Bowman",
    "Michael Barr",
    "Lisa Cook",
    "Jerome Powell",
    "Christopher Waller",
    "Susan Collins",
    "John Williams",
    "Anna Paulson",
    "Beth Hammack",
    "Thomas Barkin",
    "Cheryl Venable",
    "Austan Goolsbee",
    "Alberto Musalem",
    "Neel Kashkari",
    "Jeffrey Schmid",
    "Lorie Logan",
    "Mary Daly",
]


# ============================================================
# Member validation metadata
# ============================================================

MEMBER_VALIDATION = {
    "Kevin Warsh": {
        "last_name": "warsh",
        "anchors": [
            "federal reserve",
            "fed chair",
            "fed chairman",
            "fed chief",
        ],
    },

    "Philip Jefferson": {
        "last_name": "jefferson",
        "anchors": [
            "federal reserve",
            "fed governor",
            "fed vice chair",
            "vice chair",
        ],
    },

    "Michelle Bowman": {
        "last_name": "bowman",
        "anchors": [
            "federal reserve",
            "fed governor",
            "fed vice chair",
            "vice chair for supervision",
        ],
    },

    "Michael Barr": {
        "last_name": "barr",
        "anchors": [
            "federal reserve",
            "fed governor",
        ],
    },

    "Lisa Cook": {
        "last_name": "cook",
        "anchors": [
            "federal reserve",
            "fed governor",
        ],
    },

    "Jerome Powell": {
        "last_name": "powell",
        "anchors": [
            "federal reserve",
            "fed chair",
            "fed chairman",
        ],
    },

    "Christopher Waller": {
        "last_name": "waller",
        "anchors": [
            "federal reserve",
            "fed governor",
        ],
    },

    # 동명이인 미국 상원의원 Susan Collins가 있어서
    # Boston Fed 관련 단서를 더 엄격하게 본다.
    "Susan Collins": {
        "last_name": "collins",
        "anchors": [
            "boston fed",
            "federal reserve bank of boston",
            "boston federal reserve",
            "fed's collins",
            "fed’s collins",
        ],
    },

    "John Williams": {
        "last_name": "williams",
        "anchors": [
            "new york fed",
            "federal reserve bank of new york",
            "new york federal reserve",
            "fed's williams",
            "fed’s williams",
        ],
    },

    "Anna Paulson": {
        "last_name": "paulson",
        "anchors": [
            "philadelphia fed",
            "federal reserve bank of philadelphia",
            "philadelphia federal reserve",
            "fed's paulson",
            "fed’s paulson",
        ],
    },

    "Beth Hammack": {
        "last_name": "hammack",
        "anchors": [
            "cleveland fed",
            "federal reserve bank of cleveland",
            "cleveland federal reserve",
            "fed's hammack",
            "fed’s hammack",
        ],
    },

    "Thomas Barkin": {
        "last_name": "barkin",
        "anchors": [
            "richmond fed",
            "federal reserve bank of richmond",
            "richmond federal reserve",
            "fed's barkin",
            "fed’s barkin",
        ],
    },

    "Cheryl Venable": {
        "last_name": "venable",
        "anchors": [
            "atlanta fed",
            "federal reserve bank of atlanta",
            "atlanta federal reserve",
            "fed's venable",
            "fed’s venable",
        ],
    },

    "Austan Goolsbee": {
        "last_name": "goolsbee",
        "anchors": [
            "chicago fed",
            "federal reserve bank of chicago",
            "chicago federal reserve",
            "fed's goolsbee",
            "fed’s goolsbee",
        ],
    },

    "Alberto Musalem": {
        "last_name": "musalem",
        "anchors": [
            "st. louis fed",
            "st louis fed",
            "federal reserve bank of st. louis",
            "fed's musalem",
            "fed’s musalem",
        ],
    },

    "Neel Kashkari": {
        "last_name": "kashkari",
        "anchors": [
            "minneapolis fed",
            "federal reserve bank of minneapolis",
            "minneapolis federal reserve",
            "fed's kashkari",
            "fed’s kashkari",
        ],
    },

    "Jeffrey Schmid": {
        "last_name": "schmid",
        "anchors": [
            "kansas city fed",
            "federal reserve bank of kansas city",
            "kansas city federal reserve",
            "fed's schmid",
            "fed’s schmid",
        ],
    },

    "Lorie Logan": {
        "last_name": "logan",
        "anchors": [
            "dallas fed",
            "federal reserve bank of dallas",
            "dallas federal reserve",
            "fed's logan",
            "fed’s logan",
        ],
    },

    "Mary Daly": {
        "last_name": "daly",
        "anchors": [
            "san francisco fed",
            "federal reserve bank of san francisco",
            "san francisco federal reserve",
            "fed's daly",
            "fed’s daly",
        ],
    },
}


SOURCE_PRIORITY = [
    "www.reuters.com",
    "www.bloomberg.com",
    "www.cnbc.com",
    "www.wsj.com",
    "apnews.com",
    "finance.yahoo.com",
]


# ============================================================
# Basic
# ============================================================

def _get_api_key() -> str:
    api_key = os.getenv(
        "FINLIGHT_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "FINLIGHT_API_KEY is not set."
        )

    return api_key


def _date_range(
    lookback_days: int,
):
    end = datetime.now(
        timezone.utc
    )

    start = end - timedelta(
        days=lookback_days
    )

    return (
        start.strftime(
            "%Y-%m-%d"
        ),
        end.strftime(
            "%Y-%m-%d"
        ),
    )


def _clean_text(
    value,
):
    text = str(
        value
        or ""
    ).lower()

    text = (
        text
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# Speaker relevance
# ============================================================

def _article_text(
    article,
):
    """
    검증용 텍스트.

    full article body까지 요구하지 않고
    Finlight가 기본 제공하는 title + summary만 사용한다.
    """
    title = article.get(
        "title",
        ""
    )

    summary = article.get(
        "summary",
        ""
    )

    return _clean_text(
        f"{title} {summary}"
    )


def is_member_relevant(member, article):
    if member != "Susan Collins":
        return True

    text = _article_text(article)

    return any(
        anchor in text
        for anchor in [
            "boston fed",
            "federal reserve bank of boston",
            "boston federal reserve",
            "fed's collins",
            "fed’s collins",
        ]
    )

def filter_member_articles(
    member: str,
    articles,
):
    accepted = []
    rejected = []

    for article in articles:
        if is_member_relevant(
            member,
            article,
        ):
            accepted.append(
                article
            )

        else:
            rejected.append(
                article
            )

    return (
        accepted,
        rejected,
    )


# ============================================================
# Finlight
# ============================================================

def search_finlight(
    query: str,
    lookback_days: int = 30,
    page_size: int = 100,
):
    """
    Finlight에서 lookback_days 기간의 전 페이지를 수집한다.
    """
    api_key = _get_api_key()

    start_date, end_date = _date_range(
        lookback_days
    )

    all_articles = []
    page = 1

    while True:
        response = requests.post(
            FINLIGHT_API_URL,
            headers={
                "X-API-KEY":
                    api_key,

                "Content-Type":
                    "application/json",

                "Accept":
                    "application/json",
            },
            json={
                "query":
                    query,

                "language":
                    "en",

                "pageSize":
                    page_size,

                "page":
                    page,

                "from":
                    start_date,

                "to":
                    end_date,
            },
            timeout=20,
        )

        response.raise_for_status()
        result = response.json()

        articles = result.get(
            "articles",
            [],
        )

        if not articles:
            break

        all_articles.extend(
            articles
        )

        if len(articles) < page_size:
            break

        page += 1

    return {
        "articles":
            all_articles,
    }


def collect_member_articles(
    member: str,
    lookback_days: int = 30,
    page_size: int = 100,
):
    result = search_finlight(
        query=member,
        lookback_days=lookback_days,
        page_size=page_size,
    )

    raw_articles = result.get(
        "articles",
        [],
    )

    accepted, rejected = (
        filter_member_articles(
            member,
            raw_articles,
        )
    )

    return {
        "raw":
            raw_articles,

        "accepted":
            accepted,

        "rejected":
            rejected,
    }


# ============================================================
# Source priority
# ============================================================

def source_rank(
    source: str,
):
    source = (
        source
        or ""
    ).lower()

    try:
        return SOURCE_PRIORITY.index(
            source
        )

    except ValueError:
        return len(
            SOURCE_PRIORITY
        )


# ============================================================
# Coverage Test
# ============================================================

def coverage_test(
    lookback_days: int = 30,
    page_size: int = 100,
):
    all_rows = []

    print()
    print(
        "=" * 100
    )
    print(
        "FINLIGHT FED COVERAGE + SPEAKER VALIDATION"
    )
    print(
        "=" * 100
    )

    for index, member in enumerate(
        FED_MEMBERS,
        start=1,
    ):
        try:
            result = (
                collect_member_articles(
                    member=member,
                    lookback_days=lookback_days,
                    page_size=page_size,
                )
            )

        except Exception as exc:
            print(
                f"[{index:02d}/{len(FED_MEMBERS)}] "
                f"{member:<22} "
                f"ERROR | {exc}"
            )

            all_rows.append(
                {
                    "member":
                        member,

                    "raw":
                        [],

                    "articles":
                        [],

                    "rejected":
                        [],

                    "error":
                        str(exc),
                }
            )

            continue

        raw_articles = result[
            "raw"
        ]

        articles = result[
            "accepted"
        ]

        rejected = result[
            "rejected"
        ]

        sources = Counter(
            (
                article.get(
                    "source"
                )
                or "UNKNOWN"
            )
            for article
            in articles
        )

        reuters_count = (
            sources.get(
                "www.reuters.com",
                0,
            )
        )

        bloomberg_count = (
            sources.get(
                "www.bloomberg.com",
                0,
            )
        )

        cnbc_count = (
            sources.get(
                "www.cnbc.com",
                0,
            )
        )

        print(
            f"[{index:02d}/{len(FED_MEMBERS)}] "
            f"{member:<22} "
            f"RAW={len(raw_articles):<3} "
            f"VALID={len(articles):<3} "
            f"DROP={len(rejected):<3} "
            f"Reuters={reuters_count:<3} "
            f"Bloomberg={bloomberg_count:<3} "
            f"CNBC={cnbc_count:<3}"
        )

        all_rows.append(
            {
                "member":
                    member,

                "raw":
                    raw_articles,

                "articles":
                    articles,

                "rejected":
                    rejected,

                "error":
                    None,
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=" * 100
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 100
    )

    covered = [
        row
        for row in all_rows
        if row[
            "articles"
        ]
    ]

    uncovered = [
        row[
            "member"
        ]
        for row in all_rows
        if not row[
            "articles"
        ]
    ]

    raw_total = sum(
        len(
            row[
                "raw"
            ]
        )
        for row
        in all_rows
    )

    valid_total = sum(
        len(
            row[
                "articles"
            ]
        )
        for row
        in all_rows
    )

    rejected_total = sum(
        len(
            row[
                "rejected"
            ]
        )
        for row
        in all_rows
    )

    source_counts = Counter()

    for row in all_rows:
        for article in row[
            "articles"
        ]:
            source = (
                article.get(
                    "source"
                )
                or "UNKNOWN"
            )

            source_counts[
                source
            ] += 1

    print(
        "Members covered:",
        len(
            covered
        ),
        "/",
        len(
            FED_MEMBERS
        ),
    )

    print(
        "Raw articles:",
        raw_total,
    )

    print(
        "Validated articles:",
        valid_total,
    )

    print(
        "Rejected articles:",
        rejected_total,
    )

    if raw_total:
        print(
            "Retention:",
            f"{valid_total / raw_total:.1%}",
        )

    print()

    print(
        "Top validated sources:"
    )

    for source, count in (
        source_counts.most_common(
            15
        )
    ):
        print(
            f"  {source:<35} "
            f"{count}"
        )

    print()

    if uncovered:
        print(
            "Uncovered members:"
        )

        for member in uncovered:
            print(
                "  -",
                member,
            )

    else:
        print(
            "Uncovered members: 0"
        )

    return all_rows


# ============================================================
# Accepted Samples
# ============================================================

def print_sample_articles(
    rows,
    per_member: int = 3,
):
    print()
    print(
        "=" * 100
    )
    print(
        "VALIDATED HIGH-QUALITY SOURCE SAMPLES"
    )
    print(
        "=" * 100
    )

    for row in rows:
        member = row[
            "member"
        ]

        articles = row[
            "articles"
        ]

        if not articles:
            continue

        ranked = sorted(
            articles,
            key=lambda article: (
                source_rank(
                    article.get(
                        "source",
                        ""
                    )
                ),
                article.get(
                    "publishDate",
                    "",
                ),
            ),
        )

        print()
        print(
            f"[{member}]"
        )

        for article in ranked[
            :per_member
        ]:
            print(
                " ",
                article.get(
                    "publishDate"
                ),
                "|",
                article.get(
                    "source"
                ),
                "|",
                (
                    article.get(
                        "title"
                    )
                    or ""
                )[:110],
            )

            print(
                "    SUMMARY:",
                (
                    article.get(
                        "summary"
                    )
                    or ""
                )[:220],
            )


# ============================================================
# Rejected Samples
# ============================================================

def print_rejected_samples(
    rows,
    per_member: int = 2,
):
    print()
    print(
        "=" * 100
    )
    print(
        "REJECTED SAMPLE CHECK"
    )
    print(
        "=" * 100
    )

    for row in rows:
        member = row[
            "member"
        ]

        rejected = row[
            "rejected"
        ]

        if not rejected:
            continue

        print()
        print(
            f"[{member}]"
        )

        for article in rejected[
            :per_member
        ]:
            print(
                "  DROP |",
                article.get(
                    "source"
                ),
                "|",
                (
                    article.get(
                        "title"
                    )
                    or ""
                )[:120],
            )


# ============================================================
# MAIN
# ============================================================


# ============================================================
# Pipeline Collector
# ============================================================

def collect_finlight_news(
    member: str,
    lookback_days: int = 14,
    page_size: int = 100,
) -> list[Document]:
    """
    기존 Finlight 검색/검증 로직을 그대로 사용해서
    pipeline용 Document로 변환한다.

    일반 member는 exact-name 검색 결과를 유지하고,
    Susan Collins는 기존 최소 동명이인 filter를 그대로 사용한다.
    """

    result = collect_member_articles(
        member=member,
        lookback_days=lookback_days,
        page_size=page_size,
    )

    articles = result[
        "accepted"
    ]

    documents = []
    seen_urls = set()

    for article in articles:
        title = str(
            article.get(
                "title"
            )
            or ""
        ).strip()

        summary = str(
            article.get(
                "summary"
            )
            or ""
        ).strip()

        url = str(
            article.get(
                "link"
            )
            or ""
        ).strip()

        published_at = (
            article.get(
                "publishDate"
            )
            or article.get(
                "publishedAt"
            )
        )

        if not title or not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(
            url
        )

        publisher = str(
            article.get(
                "source"
            )
            or "unknown"
        ).strip().lower()

        documents.append(
            Document(
                source=(
                    "news_rss_finlight"
                    f"|{publisher}"
                ),
                title=title,
                url=url,
                published_at=published_at,
                speaker=member,
                text=summary,
                fetch_ok=bool(
                    summary
                ),
            )
        )

    return documents


def collect_finlight_news_for_members(
    members: list[str],
    lookback_days: int = 14,
    page_size: int = 100,
) -> list[Document]:
    """
    여러 Fed member의 Finlight 뉴스를 수집한다.
    """

    documents = []
    total = len(
        members
    )

    for index, member in enumerate(
        members,
        start=1,
    ):
        try:
            member_documents = (
                collect_finlight_news(
                    member=member,
                    lookback_days=lookback_days,
                    page_size=page_size,
                )
            )

        except Exception as exc:
            print(
                f"[FINLIGHT {index}/{total}] "
                f"{member} -> ERROR | {exc}"
            )
            continue

        print(
            f"[FINLIGHT {index}/{total}] "
            f"{member} -> "
            f"{len(member_documents)} article(s)"
        )

        documents.extend(
            member_documents
        )

    print(
        f"[FINLIGHT] total="
        f"{len(documents)}"
    )

    return documents


if __name__ == "__main__":
    rows = coverage_test(
        lookback_days=30,
        page_size=100,
    )

    print_sample_articles(
        rows,
        per_member=3,
    )

    print_rejected_samples(
        rows,
        per_member=2,
    )