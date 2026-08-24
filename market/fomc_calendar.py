import json
from datetime import date
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

OUTPUT_PATH = (
    MARKET_DIR
    / "fomc_calendar.json"
)


# ============================================================
# 2026 OFFICIAL FOMC CALENDAR
#
# Source:
# Federal Reserve Board
# Meeting calendars and information
#
# decision_date:
#     회의 마지막 날
#     Statement / policy decision 발표일
#
# has_sep:
#     Summary of Economic Projections 발표 여부
# ============================================================

FOMC_2026 = [
    {
        "meeting_start": "2026-01-27",
        "meeting_end": "2026-01-28",
        "decision_date": "2026-01-28",
        "has_sep": False,
    },
    {
        "meeting_start": "2026-03-17",
        "meeting_end": "2026-03-18",
        "decision_date": "2026-03-18",
        "has_sep": True,
    },
    {
        "meeting_start": "2026-04-28",
        "meeting_end": "2026-04-29",
        "decision_date": "2026-04-29",
        "has_sep": False,
    },
    {
        "meeting_start": "2026-06-16",
        "meeting_end": "2026-06-17",
        "decision_date": "2026-06-17",
        "has_sep": True,
    },
    {
        "meeting_start": "2026-07-28",
        "meeting_end": "2026-07-29",
        "decision_date": "2026-07-29",
        "has_sep": False,
    },
    {
        "meeting_start": "2026-09-15",
        "meeting_end": "2026-09-16",
        "decision_date": "2026-09-16",
        "has_sep": True,
    },
    {
        "meeting_start": "2026-10-27",
        "meeting_end": "2026-10-28",
        "decision_date": "2026-10-28",
        "has_sep": False,
    },
    {
        "meeting_start": "2026-12-08",
        "meeting_end": "2026-12-09",
        "decision_date": "2026-12-09",
        "has_sep": True,
    },
]


# ============================================================
# BUILD
# ============================================================

def build_fomc_calendar():

    today = date.today().isoformat()

    events = []

    for meeting in FOMC_2026:

        decision_date = meeting[
            "decision_date"
        ]

        status = (
            "PAST"
            if decision_date <= today
            else "FUTURE"
        )

        event = {
            "event_type": "FOMC",
            "year": 2026,

            "meeting_start":
                meeting["meeting_start"],

            "meeting_end":
                meeting["meeting_end"],

            "decision_date":
                decision_date,

            # event_matcher.py에서
            # strong signal과 동일한 date 필드로
            # 처리하기 위한 공통 필드
            "date":
                decision_date,

            "has_sep":
                meeting["has_sep"],

            "status":
                status,
        }

        events.append(event)


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            events,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("=" * 80)
    print("BUILD FOMC CALENDAR")
    print("=" * 80)

    for event in events:

        sep = (
            "SEP"
            if event["has_sep"]
            else "-"
        )

        print(
            f"{event['decision_date']}"
            f" | {event['status']:6}"
            f" | {sep}"
            f" | "
            f"{event['meeting_start']}"
            f" ~ "
            f"{event['meeting_end']}"
        )


    print()
    print(
        f"Total : {len(events)}"
    )

    print(
        f"Saved : {OUTPUT_PATH}"
    )

    print("=" * 80)

    return events


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_fomc_calendar()