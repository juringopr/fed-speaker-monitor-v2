from fed_speaker_monitor_v2.collectors.wire_adapter import normalize_wire_item
from fed_speaker_monitor_v2.processors.event_grouping import group_wire_events


def test_wire_normalize_and_group():
    rows = [
        normalize_wire_item(
            {
                "id": "1",
                "headline": "FED'S TEST: INFLATION IS COOLING",
                "published_at": "2026-08-14T14:30:00+00:00",
            },
            provider="TEST",
            speaker="Test Speaker",
        ),
        normalize_wire_item(
            {
                "id": "2",
                "headline": "FED'S TEST: NEEDS MORE OF THE SAME",
                "published_at": "2026-08-14T14:34:00+00:00",
            },
            provider="TEST",
            speaker="Test Speaker",
        ),
    ]

    events = group_wire_events(rows)

    assert len(events) == 1
    assert events[0]["headline_count"] == 2
    assert "INFLATION IS COOLING" in events[0]["combined_text"]
    assert "NEEDS MORE OF THE SAME" in events[0]["combined_text"]
