from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

from fed_speaker_monitor_v2.config import LOOKBACK_DAYS, REQUEST_TIMEOUT, USER_AGENT
from fed_speaker_monitor_v2.models import Document


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

GOOGLE_NEWS_PARAMS = (
    "&hl=en-US"
    "&gl=US"
    "&ceid=US:en"
)

QUERY_GROUP_SIZE = 5
LEAD_TEXT_LIMIT = 1000

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_published_at(entry) -> datetime | None:
    for field in ("published", "updated"):
        raw = entry.get(field)

        if not raw:
            continue

        try:
            dt = parsedate_to_datetime(raw)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            return dt

        except (TypeError, ValueError, OverflowError):
            pass

    return None


def _clean_html(value: str | None) -> str:
    text = str(value or "")

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _entry_text(entry) -> str:
    title = _clean_html(
        entry.get("title", "")
    )

    summary = _clean_html(
        entry.get("summary", "")
        or entry.get("description", "")
    )

    return f"{title} {summary}".lower()


def _meta_description(
    soup: BeautifulSoup,
) -> str:
    selectors = [
        ("name", "description"),
        ("property", "og:description"),
        ("name", "twitter:description"),
    ]

    for attr_name, attr_value in selectors:
        tag = soup.find(
            "meta",
            attrs={
                attr_name: attr_value,
            },
        )

        if not tag:
            continue

        value = (
            tag.get("content", "")
            or ""
        ).strip()

        if value:
            return _clean_html(value)

    return ""


def _first_paragraphs(
    soup: BeautifulSoup,
    limit: int = LEAD_TEXT_LIMIT,
) -> str:
    node = (
        soup.find("article")
        or soup.find("main")
        or soup
    )

    parts = []
    total = 0

    for paragraph in node.find_all("p"):
        text = paragraph.get_text(
            " ",
            strip=True,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            continue

        parts.append(text)
        total += len(text)

        if total >= limit:
            break

    return " ".join(parts)[:limit].strip()


def extract_news_lead(
    url: str,
    rss_summary: str = "",
) -> str:
    """
    semantic dedup용 lightweight article context.

    전체 기사 본문 대신:
    - meta description
    - article/main 첫 문단들
    만 최대 1,000자 사용.

    기사 접근 실패 시 RSS summary fallback.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        meta = _meta_description(soup)
        lead = _first_paragraphs(soup)

        combined = " ".join(
            part
            for part in (
                meta,
                lead,
            )
            if part
        ).strip()

        if combined:
            return combined[:LEAD_TEXT_LIMIT]

    except Exception:
        pass

    return (
        rss_summary
        or ""
    )[:LEAD_TEXT_LIMIT].strip()


def _member_aliases(
    member: str,
) -> tuple[str, ...]:
    normalized = re.sub(
        r"\s+",
        " ",
        member.strip().lower(),
    )

    parts = normalized.split()

    aliases = [normalized]

    if parts:
        aliases.append(parts[-1])

    return tuple(
        dict.fromkeys(aliases)
    )


def _match_members(
    members: list[str],
    text: str,
) -> list[str]:
    """기사 문맥에서 확인되는 모든 TARGET SPEAKER를 반환."""
    matched = []

    # full name first
    for member in members:
        aliases = _member_aliases(member)

        if aliases[0] in text:
            matched.append(member)

    # last name fallback
    for member in members:
        if member in matched:
            continue

        aliases = _member_aliases(member)

        if len(aliases) < 2:
            continue

        last_name = aliases[1]

        if re.search(
            rf"\b{re.escape(last_name)}\b",
            text,
            re.IGNORECASE,
        ):
            matched.append(member)

    return matched


def _chunks(
    values: list[str],
    size: int,
):
    for index in range(
        0,
        len(values),
        size,
    ):
        yield values[
            index:
            index + size
        ]


def _build_group_query(
    members: list[str],
    lookback_days: int,
) -> str:
    names = " OR ".join(
        f'"{member}"'
        for member in members
    )

    return (
        f"({names}) "
        f'("Federal Reserve" OR Fed) '
        f"when:{lookback_days}d"
    )


def _build_feed_url(
    members: list[str],
    lookback_days: int,
) -> str:
    query = _build_group_query(
        members=members,
        lookback_days=lookback_days,
    )

    return (
        f"{GOOGLE_NEWS_RSS}"
        f"?q={quote_plus(query)}"
        f"{GOOGLE_NEWS_PARAMS}"
    )


def _normalize_url(
    value: str | None,
) -> str:
    url = (
        value
        or ""
    ).strip()

    if "#" in url:
        url = url.split("#", 1)[0]

    return url.rstrip("/")


def _normalize_title(
    value: str | None,
) -> str:
    title = (
        value
        or ""
    ).lower()

    title = re.sub(
        r"\s*[-|]\s*[^-|]{2,50}$",
        "",
        title,
    )

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        title,
    )

    return re.sub(
        r"\s+",
        " ",
        title,
    ).strip()


def _deduplicate_news(
    documents: list[Document],
) -> list[Document]:
    """
    network 단계 직후의 아주 얇은 exact-ish dedup.
    semantic event dedup은 processors/dedup.py가 담당.
    """

    results = []
    seen_urls = set()
    seen_keys = set()

    for document in documents:
        url_key = _normalize_url(
            document.url
        )

        if (
            url_key
            and url_key in seen_urls
        ):
            continue

        date_key = (
            str(document.published_at)[:10]
            if document.published_at
            else ""
        )

        title_key = _normalize_title(
            document.title
        )

        content_key = (
            document.speaker or "",
            date_key,
            title_key,
        )

        if (
            title_key
            and content_key in seen_keys
        ):
            continue

        results.append(document)

        if url_key:
            seen_urls.add(url_key)

        if title_key:
            seen_keys.add(content_key)

    return results


def _collect_group(
    members: list[str],
    lookback_days: int,
) -> list[Document]:
    feed_url = _build_feed_url(
        members=members,
        lookback_days=lookback_days,
    )

    feed = feedparser.parse(
        feed_url
    )

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=lookback_days)
    )

    documents = []

    for entry in feed.entries:
        published_at = _parse_published_at(
            entry
        )

        if (
            published_at is not None
            and published_at < cutoff
        ):
            continue

        title = _clean_html(
            entry.get("title", "")
        )

        url = str(
            entry.get("link", "")
        ).strip()

        rss_summary = _clean_html(
            entry.get("summary", "")
            or entry.get("description", "")
        )

        if not title or not url:
            continue

        lead_text = extract_news_lead(
            url=url,
            rss_summary=rss_summary,
        )

        # RSS title/summary뿐 아니라 실제 기사 lead까지 speaker matching에 사용.
        # 한 기사에 여러 Fed 인사가 등장하면 speaker별 Document를 각각 만든다.
        match_text = " ".join(
            part
            for part in (
                _entry_text(entry),
                lead_text.lower(),
            )
            if part
        )

        matched_members = _match_members(
            members=members,
            text=match_text,
        )

        if not matched_members:
            continue

        for member in matched_members:
            documents.append(
                Document(
                    source="google_news",
                    title=title,
                    url=url,
                    published_at=(
                        published_at.isoformat()
                        if published_at
                        else None
                    ),
                    speaker=member,
                    text=lead_text,
                    fetch_ok=bool(lead_text),
                )
            )

    return documents


def collect_fed_news_for_members(
    members: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> list[Document]:
    members = [
        member.strip()
        for member in members
        if member
        and member.strip()
    ]

    if not members:
        return []

    documents = []

    groups = list(
        _chunks(
            members,
            QUERY_GROUP_SIZE,
        )
    )

    for index, group in enumerate(
        groups,
        start=1,
    ):
        try:
            rows = _collect_group(
                members=group,
                lookback_days=lookback_days,
            )

            print(
                f"[NEWS RSS {index}/{len(groups)}] "
                f"{', '.join(group)} "
                f"-> {len(rows)} matched"
            )

            documents.extend(rows)

        except Exception as exc:
            print(
                f"[NEWS RSS {index}/{len(groups)}] "
                f"failed: {exc}"
            )

    documents = _deduplicate_news(
        documents
    )

    print(
        f"[NEWS RSS] deduplicated total: "
        f"{len(documents)}"
    )

    return documents


def collect_fed_news(
    member: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> list[Document]:
    return collect_fed_news_for_members(
        members=[member],
        lookback_days=lookback_days,
    )