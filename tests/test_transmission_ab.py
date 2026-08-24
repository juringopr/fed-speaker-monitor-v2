import json
from pathlib import Path

from fed_speaker_monitor_v2.models import Segment
from fed_speaker_monitor_v2.llm.stance import analyze_segment
from fed_speaker_monitor_v2.llm.transmission import analyze_transmission


# ============================================================
# 설정
# ============================================================

SEGMENTS_FILE = Path(
    "fed_speaker_monitor_v2/data/results/news_segments.json"
)

TEST_AS_OF = "2026-08-20 23:59"

MAX_SEGMENTS = 10


# ============================================================
# Transmission에 전달할 driver 결정
# ============================================================

def get_transmission_driver(stance_result):

    driver = stance_result.stance_driver

    if not driver:
        return "OTHER"

    driver = str(driver).upper()

    if driver == "NOT_APPLICABLE":
        return "OTHER"

    return driver


# ============================================================
# 비교 결과 간단 분류
# ============================================================

def compare_results(stance_result, transmission_result):

    stance = stance_result.stance
    implication = transmission_result.policy_implication

    # 기존 directional stance가 transmission과 같은 방향
    if (
        stance == "HAWKISH"
        and implication == "MORE_RESTRICTIVE"
    ):
        return "ALIGNED"

    if (
        stance == "DOVISH"
        and implication == "LESS_RESTRICTIVE"
    ):
        return "ALIGNED"

    # 기존 Neutral/Irrelevant이고
    # transmission도 방향성을 만들지 않음
    if (
        stance in {"NEUTRAL", "IRRELEVANT"}
        and implication in {
            "NO_CLEAR_DIRECTION",
            "NOT_APPLICABLE",
        }
    ):
        return "ALIGNED"

    # 기존에는 방향성이 있었지만
    # transmission에서는 명확한 정책 방향이 없음
    if (
        stance in {"HAWKISH", "DOVISH"}
        and implication == "NO_CLEAR_DIRECTION"
    ):
        return "REVIEW_OVERCLASSIFICATION"

    # 기존 Neutral/Irrelevant인데
    # transmission이 새로운 방향성을 발견
    if (
        stance in {"NEUTRAL", "IRRELEVANT"}
        and implication in {
            "MORE_RESTRICTIVE",
            "LESS_RESTRICTIVE",
        }
    ):
        return "REVIEW_MISSED_SIGNAL"

    # 서로 반대 방향
    if (
        stance == "HAWKISH"
        and implication == "LESS_RESTRICTIVE"
    ):
        return "CONFLICT"

    if (
        stance == "DOVISH"
        and implication == "MORE_RESTRICTIVE"
    ):
        return "CONFLICT"

    return "REVIEW"


# ============================================================
# 메인
# ============================================================

def main():

    segments = json.loads(
        SEGMENTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # Kevin Warsh 우선
    # --------------------------------------------------------

    warsh_segments = [
        s
        for s in segments
        if s.get("speaker") == "Kevin Warsh"
    ]

    # --------------------------------------------------------
    # 다른 speaker도 일부 포함
    # --------------------------------------------------------

    other_segments = [
        s
        for s in segments
        if s.get("speaker") != "Kevin Warsh"
    ]

    selected = (
        warsh_segments[:5]
        + other_segments[:5]
    )

    selected = selected[:MAX_SEGMENTS]

    print()
    print("=" * 110)
    print("TRANSMISSION A/B TEST")
    print("=" * 110)

    print(
        f"Segments: {len(selected)}"
    )

    print(
        f"Temporary macro as-of: {TEST_AS_OF}"
    )

    print("=" * 110)

    summary = []

    for index, raw in enumerate(
        selected,
        start=1,
    ):

        segment = Segment(**raw)

        # ====================================================
        # A: 기존 stance
        # ====================================================

        stance_result = analyze_segment(
            segment
        )

        driver = get_transmission_driver(
            stance_result
        )

        # ====================================================
        # B: transmission
        # ====================================================

        transmission_result = analyze_transmission(
            text=segment.text,
            driver=driver,
            as_of=TEST_AS_OF,
        )

        comparison = compare_results(
            stance_result,
            transmission_result,
        )

        # ====================================================
        # 출력
        # ====================================================

        print()
        print("=" * 110)
        print(
            f"[{index}/{len(selected)}]"
        )
        print(
            f"SPEAKER: {segment.speaker}"
        )
        print(
            f"ID: {segment.segment_id}"
        )
        print("-" * 110)

        print("TEXT:")
        print(
            segment.text[:1500]
        )

        print()
        print("[A] EXISTING STANCE")
        print(
            "policy_relevant:",
            stance_result.policy_relevant,
        )
        print(
            "stance:",
            stance_result.stance,
        )
        print(
            "score:",
            stance_result.score,
        )
        print(
            "content_type:",
            stance_result.content_type,
        )
        print(
            "driver:",
            stance_result.stance_driver,
        )
        print(
            "policy_action:",
            stance_result.policy_action,
        )
        print(
            "speaker_evidence:",
            stance_result.speaker_evidence_type,
        )
        print(
            "text_sufficiency:",
            stance_result.text_sufficiency,
        )
        print(
            "confidence:",
            stance_result.evidence_confidence,
        )

        print()
        print("[B] TRANSMISSION")
        print(
            "driver supplied:",
            driver,
        )
        print(
            "phenomenon:",
            transmission_result.economic_phenomenon,
        )
        print(
            "macro:",
            transmission_result.macro_assessment,
        )
        print(
            "path:",
            transmission_result.transmission_path,
        )
        print(
            "implication:",
            transmission_result.policy_implication,
        )
        print(
            "confidence:",
            transmission_result.reasoning_confidence,
        )

        print()
        print(
            "[COMPARE]:",
            comparison,
        )

        summary.append(
            {
                "speaker": segment.speaker,
                "segment_id": segment.segment_id,
                "stance": stance_result.stance,
                "driver": stance_result.stance_driver,
                "implication": (
                    transmission_result.policy_implication
                ),
                "comparison": comparison,
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    for item in summary:

        print(
            f"{item['speaker']:<20} "
            f"{item['stance']:<12} "
            f"{item['driver']:<20} "
            f"{item['implication']:<20} "
            f"{item['comparison']}"
        )


if __name__ == "__main__":
    main()