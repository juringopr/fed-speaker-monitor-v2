from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Document
# ============================================================

@dataclass
class Document:
    """
    Collector가 수집한 하나의 문서.

    예:
    - Fed 공식 speech
    - Fed 공식 statement
    - 뉴스 기사
    """

    source: str
    title: str
    url: str
    published_at: Optional[str] = None

    speaker: Optional[str] = None
    text: str = ""

    # 수집 상태
    fetch_ok: bool = True


# ============================================================
# Segment
# ============================================================

@dataclass
class Segment:
    """
    Document를 LLM 분석 단위로 나눈 segment.
    """

    segment_id: str
    document_url: str
    text: str

    speaker: Optional[str] = None


# ============================================================
# Segment Stance
# ============================================================

@dataclass
class SegmentStance:
    """
    하나의 segment에 대한 LLM 판정 결과.
    """

    segment_id: str

    # 정책 관련 여부
    policy_relevant: bool

    # HAWKISH / DOVISH / NEUTRAL / IRRELEVANT
    stance: str

    # -1.0 ~ +1.0
    score: float

    # PRESCRIPTIVE / DESCRIPTIVE / MIXED / IRRELEVANT
    content_type: str

    # DIRECT / INDIRECT / NOT_APPLICABLE
    directness: str

    # FORWARD_LOOKING / BACKWARD_LOOKING /
    # MIXED / NOT_APPLICABLE
    temporal: str

    # CERTAIN / UNCERTAIN
    uncertainty: str

    # LLM이 선택한 근거
    evidence: str = ""

    # 간단한 판정 이유
    reasoning: str = ""

    # 정책 행동
    # STRONG_EASING / EASING / LEAN_EASING /
    # HOLD / DELAY_EASING / MAINTAIN_RESTRAINT /
    # LEAN_TIGHTENING / TIGHTENING /
    # UNCLEAR / NOT_APPLICABLE
    policy_action: str = "NOT_APPLICABLE"

    # 정책 방향 신호 강도
    # WEAK / MILD / MODERATE / STRONG /
    # NOT_APPLICABLE
    signal_strength: str = "NOT_APPLICABLE"

    # ========================================================
    # BIS-inspired fields
    # ========================================================

    # 기사/segment 자체의 Fed monetary-policy relevance
    # 0 = 무관
    # 5 = monetary policy가 핵심 주제
    policy_relevance_score: int = 0

    # target speaker의 견해가 어떤 형태로 존재하는지
    # DIRECT_QUOTE / ATTRIBUTED_PARAPHRASE /
    # REPORTED_POSITION / CONTEXT_ONLY / NONE
    speaker_evidence_type: str = "NONE"

    # 실제 stance 방향을 전달하는 최소 원문 phrase
    policy_bearing_phrase: str = ""

    # stance를 발생시키는 주요 정책 주제
    # INFLATION / LABOR / GROWTH /
    # FINANCIAL_CONDITIONS / INTEREST_RATES /
    # BALANCE_SHEET / POLICY_FRAMEWORK /
    # MULTIPLE / OTHER / NOT_APPLICABLE
    stance_driver: str = "NOT_APPLICABLE"

    # BIS-style 5단계 sentiment
    # DOVISH / MOSTLY_DOVISH / NEUTRAL /
    # MOSTLY_HAWKISH / HAWKISH /
    # NOT_APPLICABLE
    bis_stance: str = "NOT_APPLICABLE"

    # target-speaker stance를 뒷받침하는 근거 품질
    # HIGH / MEDIUM / LOW / NOT_APPLICABLE
    evidence_confidence: str = "NOT_APPLICABLE"

    # 현재 segment만으로 판단 가능한 정도
    # SUFFICIENT / PARTIAL / INSUFFICIENT
    text_sufficiency: str = "INSUFFICIENT"


# ============================================================
# Member Stance
# ============================================================

@dataclass
class MemberStance:
    """
    여러 SegmentStance를 한 Fed speaker 단위로 집계한 결과.
    """

    member: str

    score: float
    stance: str

    policy_segment_count: int
    irrelevant_segment_count: int

    evidence_count: int = 0

    latest_date: Optional[str] = None

    # aggregation에서 사용한 segment
    segment_results: list[SegmentStance] = field(
        default_factory=list
    )


# ============================================================
# Validation Result
# ============================================================

@dataclass
class ValidationResult:
    """
    기존 keyword 방식과 LLM 결과를 비교한 결과.
    """

    member: str

    # 기존 keyword model
    keyword_label: Optional[str] = None
    keyword_raw_score: Optional[float] = None
    keyword_normalized_score: Optional[float] = None

    # LLM
    llm_stance: Optional[str] = None
    llm_score: Optional[float] = None
    weighted_llm_score: Optional[float] = None

    llm_directional_strength: Optional[float] = None
    llm_uncertainty: Optional[str] = None

    llm_policy_segment_count: int = 0
    llm_irrelevant_segment_count: int = 0

    # validation
    validation_label: Optional[str] = None
    validation_distance: Optional[float] = None
    validation_same_direction: Optional[bool] = None

    validation_score_gap: Optional[float] = None
    validation_score_gap_abs: Optional[float] = None

    validation_quality: Optional[str] = None


# ============================================================
# Final Member Result
# ============================================================

@dataclass
class FinalMemberResult:
    """
    pipeline 최종 결과.

    aggregation/final.py에서 생성.
    """

    member: str

    evidence_count: int = 0
    rss_count: int = 0
    google_count: int = 0

    evidence_status: Optional[str] = None
    google_fallback_needed: bool = False

    keyword_score: Optional[float] = None
    keyword_stance: Optional[str] = None

    llm_score: Optional[float] = None
    llm_stance: Optional[str] = None

    validation_result: Optional[str] = None
    score_gap: Optional[float] = None

    final_score: Optional[float] = None
    final_stance: Optional[str] = None

    latest_date: Optional[str] = None