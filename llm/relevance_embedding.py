from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

RAW_DIRECT_THRESHOLD = 0.80
RAW_CANDIDATE_THRESHOLD = 0.50
NORMALIZED_RESCUE_THRESHOLD = 0.85


# ============================================================
# TEXT
# ============================================================

def _clean_text(
    value: Any,
) -> str:

    return str(
        value
        or ""
    ).strip()


def _target_text(
    row: dict,
) -> str:

    return _clean_text(
        row.get(
            "target_text",
            "",
        )
    )


def _normalized_text(
    row: dict,
) -> str:

    return _clean_text(
        row.get(
            "normalized_target_text",
            "",
        )
    )


# ============================================================
# MODEL / COSINE
# ============================================================

_EMBEDDING_MODEL = None


def _get_embedding_model() -> SentenceTransformer:

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(
            MODEL_NAME
        )

    return _EMBEDDING_MODEL


def _cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> float:

    return float(
        np.dot(
            left,
            right,
        )
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_relevance_cluster(
    rows: list[dict],
) -> list[list[int]]:
    """
    한 Raw Anchor pre-cluster 안의 Relevance PASS rows를 검증한다.

    순서:
        1. target_text Raw similarity 우선
        2. Raw >= 0.80 -> 같은 검증 후보
        3. Raw 0.50~0.80 -> normalized similarity 확인
        4. normalized >= 0.85 -> rescue
        5. 그 외 -> 분리

    이 단계에서는 Final Event를 확정하지 않는다.
    """

    valid_rows = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("passed") is True
            and _target_text(row)
            and _normalized_text(row)
        )
    ]

    if not valid_rows:
        return []

    if len(valid_rows) == 1:
        return [[0]]

    model = _get_embedding_model()

    raw_embeddings = model.encode(
        [
            _target_text(row)
            for row in valid_rows
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    normalized_embeddings = model.encode(
        [
            _normalized_text(row)
            for row in valid_rows
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    clusters = []
    anchors = []

    for current in range(
        len(valid_rows)
    ):

        best_cluster = None
        best_similarity = -1.0

        for cluster_id, anchor in enumerate(
            anchors
        ):

            raw_similarity = _cosine_similarity(
                raw_embeddings[current],
                raw_embeddings[anchor],
            )

            accepted = False
            decision_similarity = raw_similarity

            if (
                raw_similarity
                >= RAW_DIRECT_THRESHOLD
            ):
                accepted = True

            elif (
                raw_similarity
                >= RAW_CANDIDATE_THRESHOLD
            ):
                normalized_similarity = (
                    _cosine_similarity(
                        normalized_embeddings[current],
                        normalized_embeddings[anchor],
                    )
                )

                if (
                    normalized_similarity
                    >= NORMALIZED_RESCUE_THRESHOLD
                ):
                    accepted = True
                    decision_similarity = (
                        normalized_similarity
                    )

            if not accepted:
                continue

            if (
                decision_similarity
                > best_similarity
            ):
                best_similarity = (
                    decision_similarity
                )
                best_cluster = cluster_id

        if best_cluster is None:
            clusters.append(
                [current]
            )
            anchors.append(
                current
            )

        else:
            clusters[
                best_cluster
            ].append(
                current
            )

    return clusters
