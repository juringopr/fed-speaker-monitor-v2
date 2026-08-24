"""
Regional Fed official speech collector.

핵심 원칙
- 새 파일 추가 없이 이 파일 하나에서 12개 Regional Fed 처리
- 기본 수집 대상: 2026년 전체
- LOOKBACK_DAYS 사용하지 않음
- discovery 단계에서 가능한 날짜를 먼저 읽음
- 날짜가 이미 2026이 아니면 detail/body 요청 전에 즉시 제외
- 날짜가 없는 후보만 detail page에서 날짜 확인
- body fetch는 2026 후보에 대해서만 수행
- 날짜 성격을 date_type으로 구분
    SPEECH
    PUBLISHED
    RSS
    URL_DATE
    UNKNOWN
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from fed_speaker_monitor_v2.collectors.article_text import extract_article_text
from fed_speaker_monitor_v2.config import REQUEST_TIMEOUT, USER_AGENT
from fed_speaker_monitor_v2.models import Document


TARGET_YEAR = 2026

SOURCES = [
    {
        "bank": "Atlanta",
        "speaker": "Cheryl Venable",
        "mode": "rss",
        "url": "https://www.atlantafed.org/rss/speechindex",
        "kind": "atlanta",
    },
    {
        "bank": "Richmond",
        "speaker": "Thomas Barkin",
        "mode": "rss",
        "url": "https://www.richmondfed.org/press_room/speeches?cc_view=rss",
        "kind": "richmond",
    },
    {
        "bank": "Boston",
        "speaker": "Susan Collins",
        "mode": "html",
        "url": "https://www.bostonfed.org/news-and-events/speeches.aspx",
        "kind": "boston",
    },
    {
        "bank": "New York",
        "speaker": "John Williams",
        "mode": "html",
        "url": "https://www.newyorkfed.org/newsevents/speeches/index",
        "kind": "new_york",
    },
    {
        "bank": "San Francisco",
        "speaker": "Mary Daly",
        "mode": "html",
        "url": "https://www.frbsf.org/news-and-media/speeches/",
        "kind": "san_francisco",
    },
    {
        "bank": "Chicago",
        "speaker": "Austan Goolsbee",
        "mode": "js",
        "url": (
            "https://www.chicagofed.org/utilities/about-us/"
            "office-of-the-president/office-of-the-president-speaking"
        ),
        "kind": "chicago",
    },
    {
        "bank": "Cleveland",
        "speaker": "Beth Hammack",
        "mode": "js",
        "url": "https://www.clevelandfed.org/collections/speeches",
        "kind": "cleveland",
    },
    {
        "bank": "St. Louis",
        "speaker": "Alberto Musalem",
        "mode": "js",
        "url": "https://www.stlouisfed.org/from-the-president/remarks",
        "kind": "st_louis",
    },
    {
        "bank": "Philadelphia",
        "speaker": "Anna Paulson",
        "mode": "js",
        "url": "https://www.philadelphiafed.org/the-economy/monetary-policy",
        "kind": "philadelphia",
    },
    {
        "bank": "Minneapolis",
        "speaker": "Neel Kashkari",
        "mode": "js",
        "url": "https://www.minneapolisfed.org/people/neel-kashkari",
        "kind": "minneapolis",
    },
    {
        "bank": "Kansas City",
        "speaker": "Jeffrey Schmid",
        "mode": "js",
        "url": "https://www.kansascityfed.org/speeches/",
        "kind": "kansas_city",
    },
    {
        "bank": "Dallas",
        "speaker": "Lorie Logan",
        "mode": "js",
        "url": "https://www.dallasfed.org/news/speeches/logan",
        "kind": "dallas",
    },
]

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _utc_date(year, month, day):
    try:
        return datetime(
            int(year),
            int(month),
            int(day),
            tzinfo=timezone.utc,
        )
    except (ValueError, TypeError):
        return None


def _parse_long_date(text):
    text = text or ""
    match = re.search(
        r"\b("
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
        r")\s+(\d{1,2}),?\s+(20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    return _utc_date(
        match.group(3),
        MONTHS[match.group(1).lower()],
        match.group(2),
    )


def _parse_iso_date(text):
    text = text or ""
    match = re.search(
        r"\b(20\d{2})-(\d{2})-(\d{2})\b",
        text,
    )
    if not match:
        return None

    return _utc_date(
        match.group(1),
        match.group(2),
        match.group(3),
    )


def _parse_any_date(text):
    return _parse_long_date(text) or _parse_iso_date(text)


def _extract_metadata_date(soup):
    selectors = [
        ("property", "article:published_time"),
        ("name", "releaseDate"),
        ("name", "ess:publishdatetime"),
        ("name", "date"),
        ("name", "DC.date"),
        ("name", "pubdate"),
        ("itemprop", "datePublished"),
    ]

    for attr_name, attr_value in selectors:
        tag = soup.find(
            "meta",
            attrs={attr_name: attr_value},
        )
        if not tag:
            continue

        date = _parse_any_date(
            tag.get("content", "")
        )
        if date:
            return date, "PUBLISHED"

    for time_tag in soup.find_all("time"):
        value = (
            time_tag.get("datetime", "")
            or time_tag.get_text(" ", strip=True)
        )
        date = _parse_any_date(value)
        if date:
            return date, "PUBLISHED"

    return None, "UNKNOWN"


def _extract_visible_speech_date(soup):
    text = soup.get_text(" ", strip=True)

    patterns = [
        (
            r"(?:remarks|speech|address)\s+"
            r"(?:delivered|given).*?"
            r"((?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+20\d{2})"
        ),
        (
            r"(?:delivered|presented)\s+on\s+"
            r"((?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+20\d{2})"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        if not match:
            continue

        date = _parse_long_date(
            match.group(1)
        )
        if date:
            return date, "SPEECH"

    return None, "UNKNOWN"


def _request_html(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def _render_html(url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[REGIONAL] Playwright not installed.")
        return ""

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
            )
        except Exception:
            browser = p.chromium.launch(
                headless=True,
            )

        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        page = context.new_page()
        page.set_default_navigation_timeout(30000)

        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )
            page.wait_for_timeout(2500)
            return page.content()

        except Exception as exc:
            print(
                f"[REGIONAL] browser failed: "
                f"{url} | {exc}"
            )
            return ""

        finally:
            page.close()
            context.close()
            browser.close()


def _get_index_html(source):
    if source["mode"] == "js":
        return _render_html(
            source["url"]
        )

    try:
        return _request_html(
            source["url"]
        )

    except Exception as exc:
        print(
            f"[REGIONAL] {source['bank']} "
            f"requests index failed: {exc}"
        )
        return _render_html(
            source["url"]
        )


def _paragraph_text(node):
    if node is None:
        return ""

    parts = []

    for paragraph in node.find_all("p"):
        text = paragraph.get_text(
            " ",
            strip=True,
        )
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()


def _title_from_page(soup, fallback=""):
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(
            " ",
            strip=True,
        )
        if text:
            return text

    og = soup.find(
        "meta",
        attrs={"property": "og:title"},
    )

    if og and og.get("content"):
        return og["content"].strip()

    return fallback


def _fetch_pdf_text(pdf_url):
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        response = requests.get(
            pdf_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        reader = PdfReader(
            io.BytesIO(response.content)
        )

        parts = []

        for page in reader.pages:
            text = (
                page.extract_text()
                or ""
            ).strip()

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    except Exception:
        return ""


def _candidate(
    title,
    url,
    date=None,
    date_type="UNKNOWN",
):
    return {
        "title": title or "",
        "url": url,
        "date": date,
        "date_type": date_type,
    }


def _nearest_container_text(
    anchor,
    levels=4,
):
    node = anchor

    for _ in range(levels):
        if node is None:
            break

        text = node.get_text(
            " ",
            strip=True,
        )

        date = _parse_any_date(text)

        if date:
            return text, date

        node = node.parent

    return "", None


def _rss_candidates(
    source,
    target_year,
):
    rows = []

    feed = feedparser.parse(
        source["url"]
    )

    for entry in feed.entries:
        url = str(
            entry.get("link", "")
        ).strip()

        title = str(
            entry.get("title", "")
        ).strip()

        if not url:
            continue

        if source["kind"] == "richmond":
            if (
                "/press_room/speeches/"
                "thomas_i_barkin/"
                not in url
            ):
                continue

        elif source["kind"] == "atlanta":
            combined = (
                title
                + " "
                + str(entry.get("summary", ""))
                + " "
                + url
            ).lower()

            if "venable" not in combined:
                continue

        published_at = None

        for field in (
            "published",
            "updated",
        ):
            raw = entry.get(field)

            if not raw:
                continue

            try:
                published_at = parsedate_to_datetime(
                    raw
                )

                if published_at.tzinfo is None:
                    published_at = (
                        published_at.replace(
                            tzinfo=timezone.utc
                        )
                    )
                else:
                    published_at = (
                        published_at.astimezone(
                            timezone.utc
                        )
                    )

                break

            except (
                TypeError,
                ValueError,
            ):
                pass

        if (
            published_at is not None
            and published_at.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                title,
                url,
                published_at,
                (
                    "RSS"
                    if published_at
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _boston_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        if (
            "/news-and-events/speeches/"
            not in href
        ):
            continue

        url = urljoin(
            "https://www.bostonfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=6,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _new_york_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        match = re.search(
            r"/newsevents/speeches/"
            r"(20\d{2})/"
            r"wil(\d{6})",
            href,
            re.IGNORECASE,
        )

        if not match:
            continue

        ymd = match.group(2)

        date = _utc_date(
            2000 + int(ymd[:2]),
            int(ymd[2:4]),
            int(ymd[4:6]),
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        url = urljoin(
            "https://www.newyorkfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                "URL_DATE",
            )
        )

    return rows


def _san_francisco_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]
        href_lower = href.lower()

        if (
            "mary-c-daly"
            not in href_lower
            and "mary-daly"
            not in href_lower
        ):
            continue

        if not any(
            token in href_lower
            for token in (
                "/speeches/",
                "/remarks/",
                "/events/",
                "/news/",
            )
        ):
            continue

        url = urljoin(
            "https://www.frbsf.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=7,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _chicago_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        match = re.search(
            r"/publications/speeches/"
            r"(20\d{2})/"
            r"([^/?#]+)",
            href,
            re.IGNORECASE,
        )

        if not match:
            continue

        year = int(
            match.group(1)
        )

        if year != target_year:
            continue

        slug = match.group(2)

        date = None

        slug_date = re.search(
            r"(january|february|march|april|may|june|"
            r"july|august|september|october|november|december)"
            r"-(\d{1,2})",
            slug,
            re.IGNORECASE,
        )

        if slug_date:
            date = _utc_date(
                year,
                MONTHS[
                    slug_date.group(1).lower()
                ],
                int(
                    slug_date.group(2)
                ),
            )

        url = urljoin(
            "https://www.chicagofed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "URL_DATE"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _cleveland_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        match = re.search(
            r"/collections/speeches/"
            r"(?:20\d{2}/)?"
            r"sp-(20\d{6})-",
            href,
            re.IGNORECASE,
        )

        if not match:
            continue

        ymd = match.group(1)

        date = _utc_date(
            int(ymd[:4]),
            int(ymd[4:6]),
            int(ymd[6:8]),
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        url = urljoin(
            "https://www.clevelandfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                "URL_DATE",
            )
        )

    return rows


def _st_louis_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    marker = (
        f"/from-the-president/"
        f"remarks/{target_year}/"
    )

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        if marker not in href:
            continue

        url = urljoin(
            "https://www.stlouisfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=6,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _philadelphia_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        if (
            "/the-economy/"
            "monetary-policy/"
            not in href
        ):
            continue

        text = (
            anchor.get_text(
                " ",
                strip=True,
            )
            + " "
            + href
        ).lower()

        if "paulson" not in text:
            continue

        url = urljoin(
            "https://www.philadelphiafed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=7,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _minneapolis_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    marker = (
        f"/article/{target_year}/"
    )

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        if marker not in href:
            continue

        text = (
            anchor.get_text(
                " ",
                strip=True,
            )
            + " "
            + href
        ).lower()

        if not any(
            term in text
            for term in (
                "kashkari",
                "speech",
                "remarks",
            )
        ):
            continue

        url = urljoin(
            "https://www.minneapolisfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=7,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _kansas_city_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        if not re.match(
            r"^/speeches/[^/?#]+/?$",
            href,
        ):
            continue

        title = anchor.get_text(
            " ",
            strip=True,
        )

        title_lower = title.lower()

        blocked = (
            "speaker request",
            "request form",
            "speakers bureau",
            "contact",
        )

        if any(
            term in title_lower
            for term in blocked
        ):
            continue

        url = urljoin(
            "https://www.kansascityfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        _, date = _nearest_container_text(
            anchor,
            levels=7,
        )

        if (
            date is not None
            and date.year != target_year
        ):
            continue

        rows.append(
            _candidate(
                title,
                url,
                date,
                (
                    "PUBLISHED"
                    if date
                    else "UNKNOWN"
                ),
            )
        )

    return rows


def _dallas_candidates(
    soup,
    target_year,
):
    rows = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        href = anchor["href"]

        match = re.search(
            r"/news/speeches/"
            r"logan/(20\d{2})/"
            r"lkl(\d{6})",
            href,
            re.IGNORECASE,
        )

        if not match:
            continue

        year = int(
            match.group(1)
        )

        if year != target_year:
            continue

        ymd = match.group(2)

        date = _utc_date(
            2000 + int(ymd[:2]),
            int(ymd[2:4]),
            int(ymd[4:6]),
        )

        url = urljoin(
            "https://www.dallasfed.org",
            href,
        )

        if url in seen:
            continue

        seen.add(url)

        rows.append(
            _candidate(
                anchor.get_text(
                    " ",
                    strip=True,
                ),
                url,
                date,
                "URL_DATE",
            )
        )

    return rows


PARSERS = {
    "boston": _boston_candidates,
    "new_york": _new_york_candidates,
    "san_francisco": _san_francisco_candidates,
    "chicago": _chicago_candidates,
    "cleveland": _cleveland_candidates,
    "st_louis": _st_louis_candidates,
    "philadelphia": _philadelphia_candidates,
    "minneapolis": _minneapolis_candidates,
    "kansas_city": _kansas_city_candidates,
    "dallas": _dallas_candidates,
}


def _discover(
    source,
    target_year,
):
    if source["mode"] == "rss":
        return _rss_candidates(
            source,
            target_year,
        )

    html = _get_index_html(
        source
    )

    if not html:
        return []

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    parser = PARSERS[
        source["kind"]
    ]

    return parser(
        soup,
        target_year,
    )


def _detail_date_only(
    source,
    candidate,
):
    url = candidate["url"]

    if source["kind"] in {
        "st_louis",
        "chicago",
    }:
        html = _render_html(url)

    else:
        try:
            html = _request_html(url)
        except Exception:
            html = _render_html(url)

    if not html:
        return (
            None,
            "UNKNOWN",
            candidate.get("title", ""),
            None,
        )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    title = _title_from_page(
        soup,
        candidate.get(
            "title",
            "",
        ),
    )

    speech_date, _ = (
        _extract_visible_speech_date(
            soup
        )
    )

    if speech_date:
        return (
            speech_date,
            "SPEECH",
            title,
            soup,
        )

    published_date, _ = (
        _extract_metadata_date(
            soup
        )
    )

    if published_date:
        return (
            published_date,
            "PUBLISHED",
            title,
            soup,
        )

    return (
        candidate.get("date"),
        candidate.get(
            "date_type",
            "UNKNOWN",
        ),
        title,
        soup,
    )


def _fetch_body(
    source,
    candidate,
    existing_soup=None,
):
    url = candidate["url"]

    if (
        source["kind"] == "boston"
        and existing_soup is not None
    ):
        for anchor in existing_soup.find_all(
            "a",
            href=True,
        ):
            href = anchor["href"]

            if (
                href.lower()
                .split("?")[0]
                .endswith(".pdf")
            ):
                pdf_text = _fetch_pdf_text(
                    urljoin(
                        url,
                        href,
                    )
                )

                if pdf_text:
                    return pdf_text

    try:
        text = extract_article_text(
            url
        )
    except Exception:
        text = ""

    if (
        text
        and len(text) >= 300
    ):
        return text

    soup = existing_soup

    if soup is None:
        if source["kind"] in {
            "st_louis",
            "chicago",
        }:
            html = _render_html(url)

        else:
            try:
                html = _request_html(url)
            except Exception:
                html = _render_html(url)

        if not html:
            return text or ""

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

    if source["kind"] == "boston":
        for anchor in soup.find_all(
            "a",
            href=True,
        ):
            href = anchor["href"]

            if (
                href.lower()
                .split("?")[0]
                .endswith(".pdf")
            ):
                pdf_text = _fetch_pdf_text(
                    urljoin(
                        url,
                        href,
                    )
                )

                if pdf_text:
                    return pdf_text

    node = (
        soup.find("article")
        or soup.find("main")
    )

    body = _paragraph_text(
        node
    )

    return body or text or ""


def _attach_date_type(
    document,
    date_type,
):
    try:
        document.date_type = date_type
    except Exception:
        pass

    return document


def collect_regional_official(
    target_year=TARGET_YEAR,
    skip_urls=None,
):
    documents = []
    seen_urls = set()
    skip_urls = set(skip_urls or [])

    for source in SOURCES:

        try:
            candidates = _discover(
                source,
                target_year,
            )

        except Exception as exc:
            print(
                f"[REGIONAL] "
                f"{source['bank']} "
                f"discovery failed: "
                f"{exc}"
            )
            continue

        year_candidates = []
        need_date_lookup = []

        # 1) index에서 날짜가 이미 있으면 2026 선필터
        # incremental 실행 시 기존 URL은 detail/body 요청 전에 제외
        for candidate in candidates:
            if candidate.get("url") in skip_urls:
                continue

            date = candidate.get("date")

            if date is None:
                need_date_lookup.append(
                    candidate
                )
                continue

            if date.year == target_year:
                year_candidates.append({
                    **candidate,
                    "_soup": None,
                })

        # 2) 날짜 없는 후보만 detail에서 날짜 확인
        for candidate in need_date_lookup:
            if candidate.get("url") in skip_urls:
                continue

            (
                date,
                date_type,
                title,
                soup,
            ) = _detail_date_only(
                source,
                candidate,
            )

            if (
                date is None
                or date.year != target_year
            ):
                continue

            year_candidates.append({
                **candidate,
                "date": date,
                "date_type": date_type,
                "title": (
                    title
                    or candidate["title"]
                ),
                "_soup": soup,
            })

        # 3) 2026 후보만 body fetch
        body_fetch = 0
        accepted = 0

        for candidate in year_candidates:
            url = candidate["url"]

            if url in seen_urls or url in skip_urls:
                continue

            title = (
                candidate.get(
                    "title"
                )
                or ""
            ).strip()

            title_lower = title.lower()

            blocked_titles = (
                "speaker request",
                "request form",
                "speakers bureau",
                "contact",
            )

            if any(
                term in title_lower
                for term in blocked_titles
            ):
                continue

            body_fetch += 1

            body = _fetch_body(
                source,
                candidate,
                existing_soup=candidate.get(
                    "_soup"
                ),
            )

            if not body:
                continue

            date = candidate["date"]

            document = Document(
                source=(
                    "regional_"
                    + source["bank"]
                    .lower()
                    .replace(
                        " ",
                        "_",
                    )
                ),
                title=title,
                url=url,
                published_at=(
                    date.isoformat()
                ),
                speaker=source["speaker"],
                text=body,
                fetch_ok=True,
            )

            _attach_date_type(
                document,
                candidate.get(
                    "date_type",
                    "UNKNOWN",
                ),
            )

            documents.append(
                document
            )

            seen_urls.add(
                url
            )

            accepted += 1

        print(
            f"[REGIONAL] "
            f"{source['bank']}: "
            f"discovered={len(candidates)} "
            f"year_2026={len(year_candidates)} "
            f"body_fetch={body_fetch} "
            f"accepted={accepted}"
        )

    return documents