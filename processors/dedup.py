from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np


TITLE_SIMILARITY_THRESHOLD = 0.92

NEWS_SEMANTIC_THRESHOLD = 0.78
NEWS_DATE_WINDOW_DAYS = 4
NEWS_SOURCE_PREFIXES = (
    "google_news",
    "news_rss_",
)

_EMBEDDING_MODEL = None


def _clean_text(
    value: str | None,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        (value or "").strip().lower(),
    )


def _date_only(
    value: str | None,
) -> str:
    if not value:
        return ""

    return str(value)[:10]


def _parse_date(
    value: str | None,
):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        ).date()

    except ValueError:
        return None


def _normalized_url(
    value: str | None,
) -> str:
    url = (
        value
        or ""
    ).strip()

    if "#" in url:
        url = url.split(
            "#",
            1,
        )[0]

    return url.rstrip("/")


def _title_similarity(
    left: str | None,
    right: str | None,
) -> float:
    a = _clean_text(left)
    b = _clean_text(right)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def _is_news_document(
    document,
) -> bool:
    source = _clean_text(
        getattr(
            document,
            "source",
            "",
        )
    )

    return any(
        source.startswith(prefix)
        for prefix in NEWS_SOURCE_PREFIXES
    )


def _news_text(
    document,
) -> str:
    """
    title + fed_news.py에서 넣은
    meta description/article lead.
    """

    title = (
        getattr(
            document,
            "title",
            "",
        )
        or ""
    ).strip()

    text = (
        getattr(
            document,
            "text",
            "",
        )
        or ""
    ).strip()

    return (
        f"{title}. {text[:1200]}"
    ).strip()


def _get_embedding_model():
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import (
            SentenceTransformer,
        )

    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. "
            "Install with: "
            "pip install sentence-transformers"
        ) from exc

    _EMBEDDING_MODEL = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return _EMBEDDING_MODEL


def _cosine_similarity(
    left,
    right,
) -> float:
    # model.encode(normalize_embeddings=True)이므로
    # dot product가 cosine similarity와 같다.
    return float(
        np.dot(
            left,
            right,
        )
    )


def _deduplicate_basic(
    documents,
):
    """
    기존 official/document dedup.
    """

    results = []
    seen_urls = set()

    for document in documents:
        url_key = _normalized_url(
            getattr(
                document,
                "url",
                "",
            )
        )

        if (
            url_key
            and url_key in seen_urls
        ):
            continue

        duplicate = False

        speaker = _clean_text(
            getattr(
                document,
                "speaker",
                "",
            )
        )

        date = _date_only(
            getattr(
                document,
                "published_at",
                "",
            )
        )

        title = getattr(
            document,
            "title",
            "",
        )

        if speaker and date and title:
            for existing in results:
                existing_speaker = _clean_text(
                    getattr(
                        existing,
                        "speaker",
                        "",
                    )
                )

                if (
                    existing_speaker
                    != speaker
                ):
                    continue

                existing_date = _date_only(
                    getattr(
                        existing,
                        "published_at",
                        "",
                    )
                )

                if existing_date != date:
                    continue

                similarity = _title_similarity(
                    title,
                    getattr(
                        existing,
                        "title",
                        "",
                    ),
                )

                if (
                    similarity
                    >= TITLE_SIMILARITY_THRESHOLD
                ):
                    duplicate = True
                    break

        if duplicate:
            continue

        results.append(document)

        if url_key:
            seen_urls.add(url_key)

    return results


def _build_news_clusters(
    news_documents,
    embeddings,
    similarity_threshold,
    date_window_days,
):
    """
    Anchor-based event clustering.

    각 cluster의 최초 기사(anchor)와 직접 비교한다.

    같은 speaker이고,
    날짜 차이가 date_window_days 이내이며,
    anchor와 cosine similarity가 threshold 이상일 때만
    기존 cluster에 포함한다.

    Connected-component처럼
    A-B 유사 + B-C 유사만으로
    A/B/C 전체가 연결되는 chain clustering은 허용하지 않는다.
    """

    if not news_documents:
        return []

    clusters = []
    anchors = []

    for current_index, current_doc in enumerate(
        news_documents
    ):
        current_speaker = _clean_text(
            getattr(
                current_doc,
                "speaker",
                "",
            )
        )

        current_date = _parse_date(
            getattr(
                current_doc,
                "published_at",
                "",
            )
        )

        best_cluster = None
        best_similarity = -1.0

        # ----------------------------------------------------
        # 기존 cluster의 anchor와만 비교
        # ----------------------------------------------------

        for cluster_id, anchor_index in enumerate(
            anchors
        ):
            anchor_doc = news_documents[
                anchor_index
            ]

            anchor_speaker = _clean_text(
                getattr(
                    anchor_doc,
                    "speaker",
                    "",
                )
            )

            # 다른 Fed member끼리는 묶지 않음
            if (
                current_speaker
                != anchor_speaker
            ):
                continue

            anchor_date = _parse_date(
                getattr(
                    anchor_doc,
                    "published_at",
                    "",
                )
            )

            # 날짜 window 유지
            if (
                current_date is not None
                and anchor_date is not None
            ):
                day_gap = abs(
                    (
                        current_date
                        - anchor_date
                    ).days
                )

                if (
                    day_gap
                    > date_window_days
                ):
                    continue

            # Anchor와 직접 cosine similarity 비교
            similarity = _cosine_similarity(
                embeddings[
                    current_index
                ],
                embeddings[
                    anchor_index
                ],
            )

            if (
                similarity
                < similarity_threshold
            ):
                continue

            # 여러 anchor와 유사하면
            # 가장 similarity가 높은 cluster 선택
            if (
                similarity
                > best_similarity
            ):
                best_similarity = similarity
                best_cluster = cluster_id

        # ----------------------------------------------------
        # 기존 cluster에 합류
        # ----------------------------------------------------

        if best_cluster is not None:
            clusters[
                best_cluster
            ].append(
                current_index
            )

        # ----------------------------------------------------
        # 새로운 cluster 생성
        # 현재 기사가 새로운 anchor
        # ----------------------------------------------------

        else:
            clusters.append(
                [
                    current_index
                ]
            )

            anchors.append(
                current_index
            )

    return clusters

SOURCE_PRIORITY = (
    "reuters",
    "bloomberg",
    "cnbc",
    "yahoo finance",
    "financial times",
    "wall street journal",
)


def _representative_rank(
    document,
):
    title = _clean_text(
        getattr(
            document,
            "title",
            "",
        )
    )

    source = _clean_text(
        getattr(
            document,
            "source",
            "",
        )
    )

    combined = (
        f"{source} {title}"
    )

    for index, source_name in enumerate(
        SOURCE_PRIORITY
    ):
        if source_name in combined:
            return index

    return len(SOURCE_PRIORITY)


def _select_representative(
    cluster_documents,
):
    """
    같은 event에서 대표기사 1개 선택.

    1. source priority
    2. lead text가 더 긴 기사
    """

    return min(
        cluster_documents,
        key=lambda document: (
            _representative_rank(
                document
            ),
            -len(
                getattr(
                    document,
                    "text",
                    "",
                )
                or ""
            ),
        ),
    )


def deduplicate_news_semantic(
    documents,
    similarity_threshold=NEWS_SEMANTIC_THRESHOLD,
    date_window_days=NEWS_DATE_WINDOW_DAYS,
):
    news_documents = [
        document
        for document in documents
        if _is_news_document(document)
    ]

    other_documents = [
        document
        for document in documents
        if not _is_news_document(document)
    ]

    if len(news_documents) <= 1:
        return documents

    model = _get_embedding_model()

    texts = [
        _news_text(document)
        for document in news_documents
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    clusters = _build_news_clusters(
        news_documents=news_documents,
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
        date_window_days=date_window_days,
    )

    representatives = []

    for cluster in clusters:
        cluster_documents = [
            news_documents[index]
            for index in cluster
        ]

        representatives.append(
            _select_representative(
                cluster_documents
            )
        )

    print(
        f"[NEWS EVENT CLUSTER] "
        f"before={len(news_documents)} "
        f"events={len(representatives)} "
        f"removed={len(news_documents) - len(representatives)}"
    )

    return (
        other_documents
        + representatives
    )



# ============================================================
# NEWS RELEVANCE FILTER
# ============================================================

DIRECT_SPEECH_VERBS = (
    "says",
    "said",
    "warns",
    "warned",
    "sees",
    "expects",
    "expected",
    "wants",
    "wanted",
    "argues",
    "argued",
    "notes",
    "noted",
    "tells",
    "told",
    "signals",
    "signaled",
    "indicates",
    "indicated",
    "believes",
    "believed",
    "supports",
    "supported",
    "backs",
    "backed",
    "favors",
    "favours",
    "opposes",
    "urges",
    "urged",
    "calls for",
    "called for",
    "remarks",
    "speaks",
    "speech",
    "interview",
    "comments",
    "commented",
    "testimony",
)

POLICY_CONTEXT_TERMS = (
    "inflation",
    "interest rate",
    "interest rates",
    "rate cut",
    "rate cuts",
    "rate hike",
    "rate hikes",
    "monetary policy",
    "employment",
    "labor market",
    "labour market",
    "economic outlook",
    "economy",
    "fomc",
    "federal reserve",
    " fed ",
)

OBVIOUS_NEWS_NOISE_TERMS = (
    "where to watch",
    "when to watch",
    "what time",
    "live today",
    "schedule",
    "how to watch",
    "net worth",
    "biography",
    "profile of",
    "the united states v ",
    "the united states vs ",
)


def _speaker_last_name(
    document,
) -> str:
    speaker = _clean_text(
        getattr(
            document,
            "speaker",
            "",
        )
    )

    parts = speaker.split()

    return (
        parts[-1]
        if parts
        else ""
    )


def _has_direct_speech_signal(
    document,
    text: str,
) -> bool:
    """
    speaker가 실제 발언 주체인지 가볍게 확인한다.
    """

    last_name = _speaker_last_name(
        document
    )

    if not last_name:
        return False

    matches = list(
        re.finditer(
            rf"\b{re.escape(last_name)}\b",
            text,
            re.IGNORECASE,
        )
    )

    for match in matches:
        start = max(
            0,
            match.start() - 80,
        )
        end = min(
            len(text),
            match.end() + 80,
        )

        window = text[start:end]

        if any(
            verb in window
            for verb in DIRECT_SPEECH_VERBS
        ):
            return True

    if re.search(
        rf"\bfed(?:'s)?\s+{re.escape(last_name)}\b",
        text,
        re.IGNORECASE,
    ):
        return True

    return False


def _is_relevant_news_event(
    document,
) -> bool:
    """
    semantic clustering 후 대표기사에 적용하는 얇은 relevance filter.

    +2 직접 발언 신호
    +1 Fed 맥락
    +1 통화정책/거시 맥락
    -3 명백한 noise

    score >= 2 유지.
    """

    title = _clean_text(
        getattr(
            document,
            "title",
            "",
        )
    )

    lead = _clean_text(
        getattr(
            document,
            "text",
            "",
        )
    )

    combined = (
        f"{title} {lead}"
    ).strip()

    if not combined:
        return False

    score = 0

    if _has_direct_speech_signal(
        document,
        combined,
    ):
        score += 2

    if (
        "federal reserve" in combined
        or " fed " in f" {combined} "
    ):
        score += 1

    if any(
        term in combined
        for term in POLICY_CONTEXT_TERMS
    ):
        score += 1

    if any(
        term in title
        for term in OBVIOUS_NEWS_NOISE_TERMS
    ):
        score -= 3

    return score >= 2


def filter_relevant_news_events(
    documents,
):
    """
    news event 대표기사에서 relevance noise 제거.
    official documents는 그대로 유지.
    """

    kept = []
    removed = []

    for document in documents:
        if not _is_news_document(
            document
        ):
            kept.append(
                document
            )
            continue

        if _is_relevant_news_event(
            document
        ):
            kept.append(
                document
            )
        else:
            removed.append(
                document
            )

    news_before = sum(
        _is_news_document(document)
        for document in documents
    )

    news_after = sum(
        _is_news_document(document)
        for document in kept
    )

    print(
        f"[NEWS RELEVANCE] "
        f"before={news_before} "
        f"after={news_after} "
        f"removed={len(removed)}"
    )

    for document in removed[:10]:
        print(
            "    -",
            getattr(
                document,
                "speaker",
                "",
            ),
            "|",
            getattr(
                document,
                "title",
                "",
            )[:110],
        )

    return kept


def deduplicate_documents(
    documents,
    semantic_news=True,
    apply_news_relevance=True,
):
    """
    1. URL / same speaker-date-title 기본 dedup
    2. 뉴스에만 event semantic clustering
    3. 필요할 때만 뉴스 event 대표기사 relevance filter
    """

    basic_results = _deduplicate_basic(
        documents
    )

    if not semantic_news:
        return basic_results

    has_news = any(
        _is_news_document(document)
        for document in basic_results
    )

    if not has_news:
        return basic_results

    semantic_results = deduplicate_news_semantic(
        basic_results
    )

    if not apply_news_relevance:
        return semantic_results

    return filter_relevant_news_events(
        semantic_results
    )