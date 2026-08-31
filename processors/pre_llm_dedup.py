from __future__ import annotations

from fed_speaker_monitor_v2.processors.dedup import (
    _clean_text,
    _parse_date,
    _news_text,
    _get_embedding_model,
    _cosine_similarity,
    _select_representative,
)


# ============================================================
# CONFIG
# ============================================================

RAW_ANCHOR_THRESHOLD = 0.78
RAW_ANCHOR_DATE_DAYS = 4


# ============================================================
# RAW ANCHOR PRE-CLUSTER
# ============================================================

def build_raw_anchor_clusters(
    news_documents,
) -> list[list[int]]:
    """
    Relevance LLM 이전의 Raw Anchor pre-cluster.

    목적:
        같은 speaker의 뉴스 중
        같은 발언일 가능성이 있는 기사들을
        raw news text 기준으로 임시 cluster한다.

    기준:
        - same speaker
        - anchor와 published_at 차이 <= 4일
        - raw similarity >= 0.78
        - connected-component chain 허용하지 않음

    주의:
        이 단계에서는 최종 Event를 확정하지 않는다.
    """

    if not news_documents:
        return []

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

            if current_speaker != anchor_speaker:
                continue

            anchor_date = _parse_date(
                getattr(
                    anchor_doc,
                    "published_at",
                    "",
                )
            )

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

                if day_gap > RAW_ANCHOR_DATE_DAYS:
                    continue

            similarity = _cosine_similarity(
                embeddings[current_index],
                embeddings[anchor_index],
            )

            if similarity < RAW_ANCHOR_THRESHOLD:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id

        if best_cluster is not None:
            clusters[
                best_cluster
            ].append(
                current_index
            )

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


# ============================================================
# RAW ANCHOR DEDUP
# ============================================================

def deduplicate_raw_anchor(
    news_documents,
):
    """
    Raw Anchor cluster별 대표기사 1개만 남긴다.

    순서:
        Raw news documents
            ↓
        Anchor clustering
        (.78 / 4D)
            ↓
        cluster별 representative 선택
            ↓
        representative documents만 반환

    이 단계의 목적:
        Relevance LLM 이전에
        명백하게 중복되는 Raw 뉴스 표본을 줄인다.

    이 단계에서는:
        - policy relevance를 판단하지 않는다.
        - target_text를 사용하지 않는다.
        - Final Event를 확정하지 않는다.
        - Stance를 판단하지 않는다.
    """

    if not news_documents:
        return []

    clusters = build_raw_anchor_clusters(
        news_documents
    )

    representatives = []

    for cluster in clusters:

        cluster_documents = [
            news_documents[index]
            for index in cluster
        ]

        representative = (
            _select_representative(
                cluster_documents
            )
        )

        representatives.append(
            representative
        )

    print(
        "[RAW ANCHOR DEDUP] "
        f"articles={len(news_documents)} "
        f"representatives={len(representatives)} "
        f"removed="
        f"{len(news_documents) - len(representatives)}"
    )

    return representatives