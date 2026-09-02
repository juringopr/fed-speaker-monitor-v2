from __future__ import annotations

import hashlib
from collections import defaultdict
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

FINAL_EVENT_THRESHOLD = 0.85


# ============================================================
# MODEL
# ============================================================

_EMBEDDING_MODEL = None


def _get_embedding_model() -> SentenceTransformer:

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(
            MODEL_NAME
        )

    return _EMBEDDING_MODEL


# ============================================================
# TEXT / COSINE
# ============================================================

def _clean_text(
    value: Any,
) -> str:

    return str(
        value
        or ""
    ).strip()


def _normalized_text(
    row: dict,
) -> str:

    return _clean_text(
        row.get(
            "normalized_target_text",
            "",
        )
    )


def _target_text(
    row: dict,
) -> str:

    return _clean_text(
        row.get(
            "target_text",
            "",
        )
    )


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
# FINAL BEST-MATCH
# ============================================================

def _build_final_event_clusters(
    rows: list[dict],
    normalized_embeddings: np.ndarray,
) -> list[list[int]]:
    """
    같은 speaker의 ② 검증 통과 표본을 전문 전수 비교한다.

    현재 article과 기존 Final Event cluster 안의 article 중
    하나라도 normalized 전문 similarity가
    FINAL_EVENT_THRESHOLD 이상이면 같은 Final Event로 묶는다.

    여러 cluster와 매칭될 경우
    가장 높은 similarity를 가진 cluster를 선택한다.
    """

    clusters = []

    for current in range(
        len(rows)
    ):

        best_cluster = None
        best_similarity = -1.0

        for cluster_id, cluster in enumerate(
            clusters
        ):

            similarities = [
                _cosine_similarity(
                    normalized_embeddings[current],
                    normalized_embeddings[existing],
                )
                for existing in cluster
            ]

            max_similarity = max(
                similarities
            )

            if (
                max_similarity
                < FINAL_EVENT_THRESHOLD
            ):
                continue

            if (
                max_similarity
                > best_similarity
            ):
                best_similarity = (
                    max_similarity
                )
                best_cluster = cluster_id

        if best_cluster is None:
            clusters.append(
                [current]
            )

        else:
            clusters[
                best_cluster
            ].append(
                current
            )

    return clusters


# ============================================================
# REPRESENTATIVE / STABLE ID
# ============================================================

def _select_event_representative(
    cluster_rows: list[dict],
) -> dict:

    return max(
        cluster_rows,
        key=lambda row:
            len(
                _target_text(
                    row
                )
            ),
    )


def _speaker_slug(
    speaker: str,
) -> str:

    return (
        speaker
        .lower()
        .replace(
            " ",
            "_",
        )
    )


def _stable_event_id(
    speaker: str,
    segment_ids: list[str],
) -> str:

    valid_ids = sorted(
        segment_id
        for segment_id in segment_ids
        if segment_id
    )

    if not valid_ids:
        raise ValueError(
            "Cannot build event_id without segment_id."
        )

    anchor = valid_ids[0]

    digest = hashlib.sha1(
        anchor.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{_speaker_slug(speaker)}"
        f"_event_{digest}"
    )


# ============================================================
# BUILD EVENT
# ============================================================

def _build_event(
    speaker: str,
    cluster_rows: list[dict],
) -> dict:

    representative = (
        _select_event_representative(
            cluster_rows
        )
    )

    segment_ids = [
        _clean_text(
            row.get(
                "segment_id",
                "",
            )
        )
        for row in cluster_rows
    ]

    return {
        "event_id":
            _stable_event_id(
                speaker,
                segment_ids,
            ),

        "speaker":
            speaker,

        "article_count":
            len(
                cluster_rows
            ),

        "segment_ids":
            segment_ids,

        "representative_segment_id":
            _clean_text(
                representative.get(
                    "segment_id",
                    "",
                )
            ),

        "target_text":
            _target_text(
                representative
            ),

        "normalized_target_text":
            _normalized_text(
                representative
            ),

        "policy_relevance_score":
            representative.get(
                "policy_relevance_score"
            ),

        "attribution":
            representative.get(
                "attribution"
            ),

        "remark_type":
            representative.get(
                "remark_type"
            ),

        "gate_confidence":
            representative.get(
                "gate_confidence"
            ),
    }


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def build_final_events(
    validated_rows: list[dict],
) -> list[dict]:
    """
    ③ Speaker-level Final Event 전문 전수비교.

    입력:
        ② Relevance + Raw/Normalized 검증을 통과한 rows.

    순서:
        1. speaker별 재집합
        2. normalized_target_text 전문 embedding
        3. 같은 speaker 내 best-match 전문 전수 비교
        4. Final Event 생성
        5. 가장 긴 target_text를 stance용 대표문장으로 선택

    이 단계에서는 ① Raw Anchor 경계를 사용하지 않는다.
    이 단계에서는 Relevance를 다시 판정하지 않는다.
    이 단계에서는 Stance를 판정하지 않는다.
    """

    valid_rows = [
        row
        for row in validated_rows
        if (
            isinstance(row, dict)
            and row.get("passed") is True
            and _clean_text(
                row.get(
                    "speaker",
                    "",
                )
            )
            and _target_text(row)
            and _normalized_text(row)
        )
    ]

    if not valid_rows:
        return []

    speaker_groups = defaultdict(
        list
    )

    for row in valid_rows:

        speaker = _clean_text(
            row.get(
                "speaker",
                "",
            )
        )

        speaker_groups[
            speaker
        ].append(
            row
        )

    model = _get_embedding_model()

    events = []

    for speaker, rows in (
        speaker_groups.items()
    ):

        if len(rows) == 1:
            clusters = [[0]]

        else:
            normalized_embeddings = model.encode(
                [
                    _normalized_text(row)
                    for row in rows
                ],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            clusters = (
                _build_final_event_clusters(
                    rows,
                    normalized_embeddings,
                )
            )

        for cluster in clusters:

            cluster_rows = [
                rows[index]
                for index in cluster
            ]

            events.append(
                _build_event(
                    speaker,
                    cluster_rows,
                )
            )

    return events