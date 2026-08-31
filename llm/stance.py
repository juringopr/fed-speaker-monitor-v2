from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from fed_speaker_monitor_v2.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
)
from fed_speaker_monitor_v2.llm.prompt import (
    STANCE_SYSTEM_PROMPT,
    build_stance_prompt,
    NEWS_STANCE_SYSTEM_PROMPT,
    build_news_stance_prompt,
)
from fed_speaker_monitor_v2.llm.macro_background import (
    analyze_macro_background,
)
from fed_speaker_monitor_v2.models import (
    Segment,
    SegmentStance,
)


# ============================================================
# Structured Output
# ============================================================

class StanceResponse(BaseModel):
    """
    OpenAI structured output schema.

    BIS-inspired structure:
    - article policy relevance
    - target speaker attribution
    - speaker evidence
    - policy-bearing phrase
    - BIS-style stance
    - policy action
    - evidence quality
    """

    # --------------------------------------------------------
    # Article policy relevance
    # --------------------------------------------------------

    policy_relevance_score: int

    policy_relevant: bool

    # --------------------------------------------------------
    # Target speaker attribution
    # --------------------------------------------------------

    attribution: Literal[
        "TARGET_SPEAKER",
        "MIXED",
        "OTHER_SPEAKER",
        "UNCLEAR",
    ]

    speaker_evidence_type: Literal[
        "DIRECT_QUOTE",
        "ATTRIBUTED_PARAPHRASE",
        "REPORTED_POSITION",
        "CONTEXT_ONLY",
        "NONE",
    ]

    # --------------------------------------------------------
    # Policy-bearing information
    # --------------------------------------------------------

    policy_bearing_phrase: str

    stance_driver: Literal[
        "INFLATION",
        "LABOR",
        "GROWTH",
        "FINANCIAL_CONDITIONS",
        "INTEREST_RATES",
        "BALANCE_SHEET",
        "POLICY_FRAMEWORK",
        "MULTIPLE",
        "OTHER",
        "NOT_APPLICABLE",
    ]

    # --------------------------------------------------------
    # BIS-style stance
    # --------------------------------------------------------

    bis_stance: Literal[
        "DOVISH",
        "MOSTLY_DOVISH",
        "NEUTRAL",
        "MOSTLY_HAWKISH",
        "HAWKISH",
        "NOT_APPLICABLE",
    ]

    # 기존 downstream 호환용
    stance: Literal[
        "HAWKISH",
        "DOVISH",
        "NEUTRAL",
        "IRRELEVANT",
    ]

    # -1.0 ~ +1.0
    score: float

    # --------------------------------------------------------
    # Policy action
    # --------------------------------------------------------

    policy_action: Literal[
        "STRONG_EASING",
        "EASING",
        "LEAN_EASING",
        "HOLD",
        "DELAY_EASING",
        "MAINTAIN_RESTRAINT",
        "LEAN_TIGHTENING",
        "TIGHTENING",
        "UNCLEAR",
        "NOT_APPLICABLE",
    ]

    # --------------------------------------------------------
    # Signal strength
    # --------------------------------------------------------

    signal_strength: Literal[
        "WEAK",
        "MILD",
        "MODERATE",
        "STRONG",
        "NOT_APPLICABLE",
    ]

    # --------------------------------------------------------
    # Evidence quality
    # --------------------------------------------------------

    evidence_confidence: Literal[
        "HIGH",
        "MEDIUM",
        "LOW",
        "NOT_APPLICABLE",
    ]

    text_sufficiency: Literal[
        "SUFFICIENT",
        "PARTIAL",
        "INSUFFICIENT",
    ]

    # --------------------------------------------------------
    # Supporting characteristics
    # --------------------------------------------------------

    content_type: Literal[
        "PRESCRIPTIVE",
        "DESCRIPTIVE",
        "MIXED",
        "IRRELEVANT",
    ]

    directness: Literal[
        "DIRECT",
        "INDIRECT",
        "NOT_APPLICABLE",
    ]

    temporal: Literal[
        "FORWARD_LOOKING",
        "BACKWARD_LOOKING",
        "MIXED",
        "NOT_APPLICABLE",
    ]

    uncertainty: Literal[
        "CERTAIN",
        "UNCERTAIN",
    ]

    evidence: str
    reasoning: str


# ============================================================
# Client
# ============================================================

def _get_client() -> OpenAI:
    """
    OpenAI client 생성.
    """

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# Consistency
# ============================================================

def _set_irrelevant(
    result: StanceResponse,
) -> StanceResponse:
    """
    target speaker stance로 사용할 수 없는 결과를
    정규화한다.

    중요:
    article-level policy relevance는 보존한다.

    즉,
    Fed 정책 관련 뉴스이지만 target speaker의
    실제 견해가 없는 경우:

    policy_relevant = True
    stance = IRRELEVANT

    이 동시에 가능하다.
    """

    result.bis_stance = (
        "NOT_APPLICABLE"
    )

    result.stance = "IRRELEVANT"

    result.score = 0.0

    result.policy_action = (
        "NOT_APPLICABLE"
    )

    result.signal_strength = (
        "NOT_APPLICABLE"
    )

    result.evidence_confidence = (
        "NOT_APPLICABLE"
    )

    result.policy_bearing_phrase = ""

    result.content_type = (
        "IRRELEVANT"
    )

    result.directness = (
        "NOT_APPLICABLE"
    )

    result.temporal = (
        "NOT_APPLICABLE"
    )

    result.evidence = ""

    return result


def _normalize_result(
    result: StanceResponse,
) -> StanceResponse:
    """
    LLM 결과의 최소 consistency를 보정한다.

    BIS-inspired 핵심:

    1. article policy relevance와
       target-speaker stance를 분리한다.

    2. 다른 사람의 견해라고 해서
       article policy relevance를 False로 만들지 않는다.

    3. BIS-style 5단계 stance를 기존 downstream
       HAWKISH / DOVISH / NEUTRAL / IRRELEVANT
       구조로 매핑한다.

    4. 기존 aggregation과 app이 사용하는 score
       sign consistency를 유지한다.
    """

    # --------------------------------------------------------
    # 1. Policy relevance score
    # --------------------------------------------------------

    result.policy_relevance_score = max(
        0,
        min(
            5,
            result.policy_relevance_score,
        ),
    )

    # prompt의 정의와 항상 일치시킨다.
    result.policy_relevant = (
        result.policy_relevance_score >= 2
    )

    # --------------------------------------------------------
    # 2. Attribution
    # --------------------------------------------------------

    # 다른 사람의 견해이거나
    # 누구의 견해인지 불명확하면
    # target speaker stance에서는 제외.
    #
    # 단, article policy relevance는 그대로 보존.
    if result.attribution in {
        "OTHER_SPEAKER",
        "UNCLEAR",
    }:
        return _set_irrelevant(
            result
        )

    # --------------------------------------------------------
    # 3. Speaker evidence
    # --------------------------------------------------------

    # target speaker가 단순히 문맥에 등장하거나
    # 실제 견해가 없는 경우 stance 계산 제외.
    if result.speaker_evidence_type in {
        "CONTEXT_ONLY",
        "NONE",
    }:
        return _set_irrelevant(
            result
        )

    # --------------------------------------------------------
    # 4. Article policy relevance
    # --------------------------------------------------------

    # target speaker의 말이라도 monetary policy와
    # 실질적으로 관련되지 않으면 stance 계산 제외.
    if not result.policy_relevant:
        return _set_irrelevant(
            result
        )

    # --------------------------------------------------------
    # 5. BIS stance -> 기존 stance
    # --------------------------------------------------------

    if result.bis_stance in {
        "MOSTLY_HAWKISH",
        "HAWKISH",
    }:
        result.stance = "HAWKISH"

    elif result.bis_stance in {
        "MOSTLY_DOVISH",
        "DOVISH",
    }:
        result.stance = "DOVISH"

    elif result.bis_stance == "NEUTRAL":
        result.stance = "NEUTRAL"

    elif result.bis_stance == "NOT_APPLICABLE":
        return _set_irrelevant(
            result
        )

    # --------------------------------------------------------
    # 6. Score consistency
    # --------------------------------------------------------

    if result.stance == "HAWKISH":
        result.score = abs(
            result.score
        )

    elif result.stance == "DOVISH":
        result.score = -abs(
            result.score
        )

    elif result.stance == "NEUTRAL":
        result.score = 0.0

        result.signal_strength = (
            "NOT_APPLICABLE"
        )

        # Descriptive statement without an explicit policy-bearing phrase
        # must not imply a policy action.
        if (
            result.content_type == "DESCRIPTIVE"
            and not result.policy_bearing_phrase.strip()
        ):
            result.policy_action = "NOT_APPLICABLE"

        return result

    elif result.stance == "IRRELEVANT":
        return _set_irrelevant(
            result
        )

    # --------------------------------------------------------
    # 7. Score range
    # --------------------------------------------------------

    result.score = max(
        -1.0,
        min(
            1.0,
            result.score,
        ),
    )

    return result


# ============================================================
# Cache
# ============================================================

def _load_cache(
    cache_path: Path | None,
) -> dict[str, SegmentStance]:
    """
    기존 segment_stance.json을 읽어서
    segment_id -> SegmentStance 형태로 반환.

    파일이 없거나 일부 row가 깨져 있으면
    해당 row만 건너뛴다.

    새 BIS 필드는 models.py에서 default 값을 가지므로
    기존 cache도 읽을 수 있다.
    """

    if (
        cache_path is None
        or not cache_path.exists()
    ):
        return {}

    try:
        raw = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    cache = {}

    if not isinstance(
        raw,
        list,
    ):
        return cache

    for row in raw:
        if not isinstance(
            row,
            dict,
        ):
            continue

        try:
            result = SegmentStance(
                **row
            )

        except TypeError:
            continue

        cache[
            result.segment_id
        ] = result

    return cache


def _save_cache(
    cache_path: Path | None,
    results: list[SegmentStance],
):
    """
    새 LLM 결과가 하나 생길 때마다 저장한다.

    실행이 중간에 끊겨도 다음 실행에서
    이미 완료된 segment는 다시 호출하지 않는다.
    """

    if cache_path is None:
        return

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        json.dumps(
            [
                asdict(
                    result
                )
                for result
                in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Optional macro background calibration
# ============================================================

def _build_macro_background_prompt(
    as_of: str | None,
    macro_driver: str | None,
) -> str:
    """
    발언 시점의 macro background를 stance 판정의 보조 context로만 제공한다.

    중요:
    - as_of 또는 macro_driver가 없으면 기존 stance.py와 동일하게 동작한다.
    - driver를 이 함수에서 추정하지 않는다.
    - macro context 존재 여부는 데이터 존재 여부만 의미한다.
    - macro background는 speaker stance의 증거가 아니다.
    """

    if not as_of or not macro_driver:
        return ""

    result = analyze_macro_background(
        driver=macro_driver,
        as_of=as_of,
    )

    macro_context_available = (
        "No relevant macroeconomic context available"
        not in result.economic_condition
    )

    if not macro_context_available:
        return ""

    return f"""

============================================================
MACROECONOMIC BACKGROUND
============================================================

The following information describes the macroeconomic environment
available at the time of the statement.

Use this information only as calibration context for interpreting
the speaker's statement.

It may help determine whether the speaker is describing economic
conditions that were broadly consistent with, stronger than, or
weaker than the contemporaneous data.

IMPORTANT:

- macro_context_available=True means relevant macro data exists.
  It does NOT mean the speaker's statement is policy relevant.
- The macroeconomic background is NOT evidence of the speaker's stance.
- Never infer HAWKISH or DOVISH from the macroeconomic background alone.
- The speaker's own words remain the primary evidence.
- Do not assign a policy direction unless the speaker's language
  supports that direction.
- If the statement is merely descriptive, macroeconomic background
  must not turn it into a prescriptive policy signal.
- Macro background must not increase attribution confidence.
- Macro background must not substitute for a missing policy-bearing phrase.

MACRO BACKGROUND:

macro_context_available=True
stance_driver={macro_driver}
economic_condition={result.economic_condition}
recent_direction={result.recent_direction}
key_evidence={result.key_evidence}
mixed_signals={result.mixed_signals}
confidence={result.confidence}
"""


# ============================================================
# Single Segment
# ============================================================

def analyze_segment(
    segment: Segment,
    as_of: str | None = None,
    macro_driver: str | None = None,
    use_macro_background: bool = True,
) -> SegmentStance:
    """
    하나의 Segment를 OpenAI로 분석한다.

    한 번의 LLM 호출에서:

    1. article policy relevance
    2. target speaker attribution
    3. speaker evidence type
    4. policy-bearing phrase
    5. stance driver
    6. BIS-style stance
    7. policy action
    8. signal strength
    9. evidence confidence
    10. text sufficiency
    11. supporting characteristics

    를 판정한다.
    """

    client = _get_client()

    user_prompt = build_stance_prompt(
        speaker=segment.speaker,
        text=segment.text,
    )

    # A/B test용 switch.
    # False이면 macro background를 prompt에 추가하지 않는다.
    if use_macro_background:
        user_prompt += _build_macro_background_prompt(
            as_of=as_of,
            macro_driver=macro_driver,
        )

    response = client.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": STANCE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=StanceResponse,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError(
            f"Could not parse LLM result: "
            f"{segment.segment_id}"
        )

    result = _normalize_result(
        result
    )

    return SegmentStance(
        segment_id=segment.segment_id,

        # 기존 필드
        policy_relevant=result.policy_relevant,
        stance=result.stance,
        score=result.score,

        content_type=result.content_type,
        directness=result.directness,
        temporal=result.temporal,
        uncertainty=result.uncertainty,

        evidence=result.evidence,
        reasoning=result.reasoning,

        policy_action=result.policy_action,
        signal_strength=result.signal_strength,

        # BIS-inspired 필드
        policy_relevance_score=(
            result.policy_relevance_score
        ),

        speaker_evidence_type=(
            result.speaker_evidence_type
        ),

        policy_bearing_phrase=(
            result.policy_bearing_phrase
        ),

        stance_driver=(
            result.stance_driver
        ),

        bis_stance=(
            result.bis_stance
        ),

        evidence_confidence=(
            result.evidence_confidence
        ),

        text_sufficiency=(
            result.text_sufficiency
        ),
    )


# ============================================================
# Multiple Segments + Cache
# ============================================================

def analyze_segments(
    segments: list[Segment],
    cache_path: Path | None = None,
) -> list[SegmentStance]:
    """
    여러 Segment를 순차 분석한다.

    cache_path가 있으면:
    - 기존 segment_id는 API 호출하지 않음
    - 신규 segment만 LLM 호출
    - 결과는 현재 segments 순서대로 반환
    - 새 결과는 호출 직후 cache에 저장

    기존 cache 처리 방식은 유지한다.
    """

    cached = _load_cache(
        cache_path
    )

    current_ids = {
        segment.segment_id
        for segment in segments
    }

    # 현재 실행에 필요한 cache만 유지.
    # 과거에 사라진 segment 결과가 aggregation에
    # 섞이지 않게 한다.
    cached = {
        segment_id: result
        for segment_id, result
        in cached.items()
        if segment_id in current_ids
    }

    total = len(
        segments
    )

    cached_count = sum(
        1
        for segment in segments
        if segment.segment_id in cached
    )

    new_count = (
        total
        - cached_count
    )

    print(
        f"[LLM CACHE] "
        f"cached={cached_count} | "
        f"new={new_count} | "
        f"total={total}"
    )

    results_by_id = dict(
        cached
    )

    new_done = 0

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        cached_result = results_by_id.get(
            segment.segment_id
        )

        if cached_result is not None:
            print(
                f"[LLM {index}/{total}] "
                f"{segment.speaker or 'Unknown'} | "
                f"{segment.segment_id} "
                f"-> CACHE"
            )

            continue

        print(
            f"[LLM {index}/{total}] "
            f"{segment.speaker or 'Unknown'} | "
            f"{segment.segment_id}"
        )

        result = analyze_segment(
            segment
        )

        results_by_id[
            segment.segment_id
        ] = result

        new_done += 1

        print(
            f"          -> "
            f"{result.bis_stance} | "
            f"{result.stance} "
            f"{result.score:+.2f} | "
            f"relevance="
            f"{result.policy_relevance_score}/5 | "
            f"confidence="
            f"{result.evidence_confidence}"
        )

        # 현재 segment 순서 기준으로 cache 저장
        partial_results = [
            results_by_id[
                current_segment.segment_id
            ]
            for current_segment in segments
            if current_segment.segment_id
            in results_by_id
        ]

        _save_cache(
            cache_path,
            partial_results,
        )

    results = [
        results_by_id[
            segment.segment_id
        ]
        for segment in segments
    ]

    # 최종 정렬 상태로 한 번 더 저장
    _save_cache(
        cache_path,
        results,
    )

    print(
        f"[LLM CACHE] "
        f"reused={cached_count} | "
        f"analyzed={new_done}"
    )

    return results

# ============================================================
# News Macro Calibration
# ============================================================

NEWS_MACRO_SCORE_THRESHOLD = 0.40

MACRO_SUPPORTED_DRIVERS = {
    "INFLATION",
    "LABOR",
    "GROWTH",
    "FINANCIAL_CONDITIONS",
}


# ============================================================
# News Final Event - Single
# ============================================================

def _news_response_to_segment_stance(
    segment: Segment,
    result: StanceResponse,
    *,
    macro_calibrated: bool = False,
    macro_background: str = "",
) -> SegmentStance:
    """
    News StanceResponse -> SegmentStance 변환.
    """

    return SegmentStance(
        segment_id=segment.segment_id,

        policy_relevant=result.policy_relevant,
        stance=result.stance,
        score=result.score,

        content_type=result.content_type,
        directness=result.directness,
        temporal=result.temporal,
        uncertainty=result.uncertainty,

        evidence=result.evidence,
        reasoning=result.reasoning,

        policy_action=result.policy_action,
        signal_strength=result.signal_strength,

        policy_relevance_score=(
            result.policy_relevance_score
        ),

        speaker_evidence_type=(
            result.speaker_evidence_type
        ),

        policy_bearing_phrase=(
            result.policy_bearing_phrase
        ),

        stance_driver=(
            result.stance_driver
        ),

        bis_stance=(
            result.bis_stance
        ),

        evidence_confidence=(
            result.evidence_confidence
        ),

        text_sufficiency=(
            result.text_sufficiency
        ),

        macro_calibrated=macro_calibrated,
        macro_background=macro_background,
    )


def analyze_news_event(
    segment: Segment,
    as_of: str | None = None,
) -> SegmentStance:
    """
    News Final Event 전용 stance.

    1차:
        target_text만으로 News Stance 판정.

    2차:
        abs(score) < 0.40 이고
        macro 지원 driver이며
        as_of가 있는 경우에만
        당시 macro background를 추가해 재판정한다.

    Macro는 새로운 Hawk/Dove 방향을 경제지표만으로
    만들어내는 근거가 아니라, 약한 1차 방향성을
    calibration하기 위한 background context다.
    """

    client = _get_client()

    # --------------------------------------------------------
    # 1. FIRST-PASS NEWS STANCE
    # --------------------------------------------------------

    user_prompt = build_news_stance_prompt(
        speaker=segment.speaker,
        text=segment.text,
    )

    response = client.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content":
                    NEWS_STANCE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    user_prompt,
            },
        ],
        text_format=StanceResponse,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError(
            f"Could not parse News stance result: "
            f"{segment.segment_id}"
        )

    result = _normalize_result(
        result
    )

    # --------------------------------------------------------
    # 2. MACRO CALIBRATION GATE
    # --------------------------------------------------------

    should_calibrate = (
        abs(result.score)
        < NEWS_MACRO_SCORE_THRESHOLD
        and result.stance_driver
        in MACRO_SUPPORTED_DRIVERS
        and bool(as_of)
    )

    if not should_calibrate:
        return _news_response_to_segment_stance(
            segment,
            result,
        )

    macro_prompt = _build_macro_background_prompt(
        as_of=as_of,
        macro_driver=result.stance_driver,
    )

    # 관련 macro data가 없으면 1차 결과 유지.
    if not macro_prompt:
        return _news_response_to_segment_stance(
            segment,
            result,
        )

    # --------------------------------------------------------
    # 3. SECOND-PASS WITH MACRO BACKGROUND
    # --------------------------------------------------------

    calibrated_user_prompt = (
        build_news_stance_prompt(
            speaker=segment.speaker,
            text=segment.text,
        )
        + macro_prompt
        + """

============================================================
MACRO CALIBRATION INSTRUCTION
============================================================

The first-pass stance direction was weak.

Use the macroeconomic background only to calibrate how the TARGET
SPEAKER'S own language should be interpreted relative to the economic
conditions available at the time.

The macroeconomic background is NOT independent evidence of a
HAWKISH or DOVISH stance.

Do not create a directional stance from the macro data alone.

The TARGET SPEAKER'S attributed language must still support the final
direction.

If the speaker's own language remains non-directional after considering
the background, return NEUTRAL.

Do not cite macroeconomic data as the policy-bearing phrase or as
speaker evidence.

Return JSON only.
"""
    )

    calibrated_response = (
        client.responses.parse(
            model=LLM_MODEL,
            input=[
                {
                    "role": "system",
                    "content":
                        NEWS_STANCE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        calibrated_user_prompt,
                },
            ],
            text_format=StanceResponse,
        )
    )

    calibrated_result = (
        calibrated_response.output_parsed
    )

    if calibrated_result is None:
        raise RuntimeError(
            f"Could not parse calibrated News stance result: "
            f"{segment.segment_id}"
        )

    calibrated_result = _normalize_result(
        calibrated_result
    )

    # 화면에서는 macro_calibrated=True인 HAWK/DOVE를
    # "(예상) HAWKISH / (예상) DOVISH"로 표시할 수 있다.
    return _news_response_to_segment_stance(
        segment,
        calibrated_result,
        macro_calibrated=True,
        macro_background=macro_prompt,
    )


# ============================================================
# News Final Events + Cache
# ============================================================

def analyze_news_events(
    segments: list[Segment],
    cache_path: Path | None = None,
    as_of_by_id: dict[str, str] | None = None,
) -> list[SegmentStance]:
    """
    News Final Event 전용 batch stance.

    기존 analyze_segments()와 동일하게:
        - segment_id 기준 cache 재사용
        - 신규 event만 LLM 호출
        - 현재 event 순서대로 반환
        - 신규 결과 직후 cache 저장

    차이:
        - analyze_news_event() 사용
        - News 전용 prompt 사용
        - upstream Relevance gate를 다시 수행하지 않음
    """

    cached = _load_cache(
        cache_path
    )

    current_ids = {
        segment.segment_id
        for segment in segments
    }

    cached = {
        segment_id: result
        for segment_id, result
        in cached.items()
        if segment_id in current_ids
    }

    total = len(
        segments
    )

    cached_count = sum(
        1
        for segment in segments
        if segment.segment_id in cached
    )

    new_count = (
        total
        - cached_count
    )

    print(
        f"[NEWS STANCE CACHE] "
        f"cached={cached_count} | "
        f"new={new_count} | "
        f"total={total}"
    )

    results_by_id = dict(
        cached
    )

    new_done = 0

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        cached_result = (
            results_by_id.get(
                segment.segment_id
            )
        )

        if cached_result is not None:
            print(
                f"[NEWS STANCE {index}/{total}] "
                f"{segment.speaker or 'Unknown'} | "
                f"{segment.segment_id} "
                f"-> CACHE"
            )
            continue

        print(
            f"[NEWS STANCE {index}/{total}] "
            f"{segment.speaker or 'Unknown'} | "
            f"{segment.segment_id}"
        )

        as_of = None

        if as_of_by_id is not None:
            as_of = as_of_by_id.get(
                segment.segment_id
            )

        result = analyze_news_event(
            segment,
            as_of=as_of,
        )

        results_by_id[
            segment.segment_id
        ] = result

        new_done += 1

        print(
            f"          -> "
            f"{result.bis_stance} | "
            f"{result.stance} "
            f"{result.score:+.2f} | "
            f"action={result.policy_action} | "
            f"strength={result.signal_strength} | "
            f"confidence="
            f"{result.evidence_confidence}"
        )

        partial_results = [
            results_by_id[
                current_segment.segment_id
            ]
            for current_segment in segments
            if current_segment.segment_id
            in results_by_id
        ]

        _save_cache(
            cache_path,
            partial_results,
        )

    results = [
        results_by_id[
            segment.segment_id
        ]
        for segment in segments
    ]

    _save_cache(
        cache_path,
        results,
    )

    print(
        f"[NEWS STANCE CACHE] "
        f"reused={cached_count} | "
        f"analyzed={new_done}"
    )

    return results

