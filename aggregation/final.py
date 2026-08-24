from __future__ import annotations

from collections import Counter

from fed_speaker_monitor_v2.models import Document, MemberStance, FinalMemberResult


def build_final_results(
    documents: list[Document],
    member_results: list[MemberStance],
) -> list[FinalMemberResult]:
    """MemberStance를 기존 FinalMemberResult 형식으로 최소 변환한다."""
    docs_by_member = {}
    for document in documents:
        if document.speaker:
            docs_by_member.setdefault(document.speaker, []).append(document)

    final_results = []

    for member_result in member_results:
        member_docs = docs_by_member.get(member_result.member, [])
        sources = Counter(document.source for document in member_docs)

        rss_count = sum(
            count for source, count in sources.items()
            if source.startswith("fed_") or "fed_" in source
        )
        google_count = sources.get("google_news", 0)

        if member_result.policy_segment_count > 0:
            evidence_status = "POLICY_EVIDENCE"
        elif member_docs:
            evidence_status = "NO_POLICY_EVIDENCE"
        else:
            evidence_status = "NO_DOCUMENT"

        final_results.append(
            FinalMemberResult(
                member=member_result.member,
                evidence_count=member_result.evidence_count,
                rss_count=rss_count,
                google_count=google_count,
                evidence_status=evidence_status,
                google_fallback_needed=(rss_count == 0),
                llm_score=member_result.score,
                llm_stance=member_result.stance,
                final_score=member_result.score,
                final_stance=member_result.stance,
                latest_date=member_result.latest_date,
            )
        )

    return final_results
