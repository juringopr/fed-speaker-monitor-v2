from __future__ import annotations

from collections import defaultdict
from statistics import mean

from fed_speaker_monitor_v2.config import HAWKISH_THRESHOLD, DOVISH_THRESHOLD
from fed_speaker_monitor_v2.models import Document, Segment, SegmentStance, MemberStance


def _score_to_stance(score: float) -> str:
    if score >= HAWKISH_THRESHOLD:
        return "HAWKISH"
    if score <= DOVISH_THRESHOLD:
        return "DOVISH"
    return "NEUTRAL"


def aggregate_member_stances(
    documents: list[Document],
    segments: list[Segment],
    stance_results: list[SegmentStance],
) -> list[MemberStance]:
    """
    단순화된 member aggregation.

    - policy_relevant=False segment는 점수에서 제외
    - 남은 segment score를 단순 평균
    - 별도 keyword/가중치 없음
    """
    segment_map = {segment.segment_id: segment for segment in segments}
    document_map = {document.url: document for document in documents}
    grouped = defaultdict(list)

    for result in stance_results:
        segment = segment_map.get(result.segment_id)
        member = (segment.speaker if segment else None) or "Unknown"
        grouped[member].append(result)

    output = []

    for member, results in grouped.items():
        policy_results = [r for r in results if r.policy_relevant]
        irrelevant_count = len(results) - len(policy_results)
        score = mean(r.score for r in policy_results) if policy_results else 0.0
        score = round(float(score), 4)

        latest_dates = []
        for result in results:
            segment = segment_map.get(result.segment_id)
            document = document_map.get(segment.document_url) if segment else None
            if document and document.published_at:
                latest_dates.append(document.published_at)

        output.append(
            MemberStance(
                member=member,
                score=score,
                stance=_score_to_stance(score) if policy_results else "IRRELEVANT",
                policy_segment_count=len(policy_results),
                irrelevant_segment_count=irrelevant_count,
                evidence_count=sum(1 for r in policy_results if r.evidence.strip()),
                latest_date=max(latest_dates) if latest_dates else None,
                segment_results=results,
            )
        )

    return sorted(output, key=lambda item: item.member)
