from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser

from fed_speaker_monitor_v2.config import LOOKBACK_DAYS
from fed_speaker_monitor_v2.models import Document


# ============================================================
# Official Federal Reserve RSS feeds
# ============================================================

FED_OFFICIAL_FEEDS = {
    "speech": "https://www.federalreserve.gov/feeds/speeches.xml",
    "testimony": "https://www.federalreserve.gov/feeds/testimony.xml",
}


# ============================================================
# Date
# ============================================================

def _parse_published_at(entry) -> datetime | None:
    """
    RSS entry의 published date를 datetime으로 변환.
    """

    published = entry.get("published")

    if not published:
        return None

    try:
        dt = parsedate_to_datetime(published)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    except (TypeError, ValueError):
        return None


# ============================================================
# Single Feed
# ============================================================

def _collect_feed(
    source: str,
    feed_url: str,
    lookback_days: int,
) -> list[Document]:

    feed = feedparser.parse(feed_url)

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=lookback_days
    )

    documents = []

    for entry in feed.entries:

        published_at = _parse_published_at(entry)

        if published_at and published_at < cutoff:
            continue

        title = entry.get("title", "").strip()
        url = entry.get("link", "").strip()

        if not title or not url:
            continue

        documents.append(
            Document(
                source=f"fed_{source}",
                title=title,
                url=url,
                published_at=(
                    published_at.isoformat()
                    if published_at
                    else None
                ),
            )
        )

    return documents


# ============================================================
# Public Collector
# ============================================================

def collect_fed_official(
    lookback_days: int = LOOKBACK_DAYS,
) -> list[Document]:
    """
    Federal Reserve 공식 RSS에서
    speech / testimony metadata를 수집한다.

    실제 speech 본문은 fed_speeches.py에서 수집한다.
    """

    documents = []

    for source, feed_url in FED_OFFICIAL_FEEDS.items():

        feed_documents = _collect_feed(
            source=source,
            feed_url=feed_url,
            lookback_days=lookback_days,
        )

        documents.extend(feed_documents)

    return documents