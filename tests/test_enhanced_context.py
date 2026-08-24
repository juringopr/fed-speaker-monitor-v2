from fed_speaker_monitor_v2.llm.context import analyze_context
from fed_speaker_monitor_v2.validation.score_fusion import validate_score


def fake_llm(_prompt):
    return """
    {
      "relevant": true,
      "direction": "DOVISH",
      "forward_looking": true,
      "intensity": 0.7,
      "reason": "Supports less policy restraint."
    }
    """


def test_context_score():
    result = analyze_context(
        "Further progress would allow less policy restraint.",
        fake_llm,
        speaker="Example Speaker",
    )
    assert result.relevant is True
    assert result.forward_looking is True
    assert result.score == -0.7


def test_validator_disagreement_flags_review():
    result = validate_score(
        -0.7,
        {"direction": "HAWKISH", "confidence": 0.9},
    )
    assert result["review"] is True
