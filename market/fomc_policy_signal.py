from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CALENDAR_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_calendar.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_policy_signals.json"
)

FED_BASE_URL = "https://www.federalreserve.gov"


# ============================================================
# Previous-year baseline
# ============================================================

PREVIOUS_FOMC_DATE = "2025-12-10"


# ============================================================
# JSON
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Statement URL
# ============================================================

def get_statement_url(date: str) -> str:
    """
    FOMC statement URL pattern:

    https://www.federalreserve.gov/newsevents/
    pressreleases/monetaryYYYYMMDDa.htm
    """

    yyyymmdd = date.replace("-", "")

    return (
        f"{FED_BASE_URL}"
        f"/newsevents/pressreleases/"
        f"monetary{yyyymmdd}a.htm"
    )


# ============================================================
# Fetch statement
# ============================================================

def fetch_statement(date: str):
    url = get_statement_url(date)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        article = (
            soup.select_one("#article")
            or soup.select_one(
                ".col-xs-12.col-sm-8.col-md-8"
            )
            or soup.select_one("main")
        )

        if not article:
            return None

        text = article.get_text(
            "\n",
            strip=True,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return {
            "url": url,
            "text": text,
        }

    except requests.RequestException:
        return None


# ============================================================
# LLM
# ============================================================

def analyze_statement(
    statement_text: str,
    previous_statement_text: str | None = None,
):
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        )
    )

    previous_section = ""

    if previous_statement_text:
        previous_section = f"""
PREVIOUS FOMC STATEMENT:

{previous_statement_text}
"""

    prompt = f"""
You are analyzing an official Federal Open Market Committee
monetary policy statement.

Your task is to classify TWO DIFFERENT THINGS:

1. The actual policy action taken at this meeting.
2. The policy bias / stance communicated by the majority
   of the FOMC.

These are NOT the same thing.

A HOLD decision can be:

- HAWKISH HOLD
- NEUTRAL HOLD
- DOVISH HOLD

A rate cut can also contain hawkish communication.

A rate hike can also contain dovish communication.

Do NOT automatically classify HOLD as NEUTRAL.


============================================================
1. POLICY ACTION
============================================================

Determine the actual policy action.

policy_action must be one of:

- TIGHTENING
- HOLD
- EASING
- OTHER


Examples:

Rate increased:
    TIGHTENING

Rate unchanged:
    HOLD

Rate reduced:
    EASING


============================================================
2. POLICY STANCE
============================================================

Separately determine the directional policy bias
communicated by the MAJORITY of the Committee.

stance must be one of:

- HAWKISH
- NEUTRAL
- DOVISH


HAWKISH means that, relative to a genuinely balanced stance,
the statement indicates a meaningful bias toward:

- maintaining restrictive policy for longer,
- delaying expected easing,
- greater willingness to tighten,
- renewed tightening if inflation does not improve,
- materially stronger emphasis on upside inflation risks,
- reduced willingness to cut rates,
- or other policy-relevant guidance implying tighter
  future policy.

Therefore:

HOLD + HAWKISH stance = HAWKISH HOLD.


DOVISH means that the statement indicates a meaningful bias
toward:

- future easing,
- greater willingness to cut,
- increased concern about downside employment/activity risks,
- reduced concern about inflation,
- tolerance for easier financial conditions,
- or other policy-relevant guidance implying easier
  future policy.

Therefore:

HOLD + DOVISH stance = DOVISH HOLD.


NEUTRAL means:

- the Committee holds rates without a meaningful directional
  bias,
- guidance remains genuinely balanced,
- both sides of the dual mandate receive similar weight,
- the Committee preserves optionality,
- or there is insufficient policy-bearing evidence to infer
  either a hawkish or dovish bias.


============================================================
IMPORTANT: ECONOMIC DESCRIPTION IS NOT POLICY GUIDANCE
============================================================

Do NOT classify the statement as HAWKISH merely because it says:

- inflation remains elevated,
- inflation is above 2 percent,
- economic activity remains solid.

These may simply describe economic conditions.


Do NOT classify the statement as DOVISH merely because it says:

- employment has weakened,
- unemployment has risen,
- economic activity has slowed.

These may simply describe economic conditions.


Economic descriptions matter ONLY when they alter or reveal
the Committee's POLICY REACTION FUNCTION.


============================================================
POLICY-BEARING EVIDENCE
============================================================

Give the greatest weight to:

1. Actual policy decision

2. Explicit forward guidance

3. Changes in forward guidance

4. Language indicating willingness or reluctance
   to tighten or ease

5. Changes in the balance of risks

6. Material changes in the Committee's reaction function

7. Language about the expected future path of policy


Give substantially LESS weight to:

- generic inflation descriptions
- generic labor-market descriptions
- standard data-dependence language
- boilerplate dual-mandate language


============================================================
DISSENTS
============================================================

Analyze dissents separately.

dissent must be one of:

- HAWKISH
- DOVISH
- MIXED
- NONE


Examples:

A dissenter wanted a higher policy rate:
    HAWKISH

A dissenter wanted a lower policy rate:
    DOVISH

Dissenters wanted moves in both directions:
    MIXED

No dissent:
    NONE


IMPORTANT:

A dissent does NOT automatically determine the majority
Committee stance.

For example:

Majority holds rates
+
one member wants a rate cut

does NOT automatically mean the FOMC statement is dovish.

The stance field must represent the policy signal of the
MAJORITY.


============================================================
COMPARISON WITH PREVIOUS STATEMENT
============================================================

If a previous FOMC statement is provided, compare the
CURRENT statement with it.

statement_change must be one of:

- MORE_HAWKISH
- MORE_DOVISH
- UNCHANGED
- NO_COMPARISON


This is a RELATIVE measure.

It is different from stance.


Example:

Previous:
    strongly dovish

Current:
    mildly dovish

Then:

stance = DOVISH
statement_change = MORE_HAWKISH


Another example:

Previous:
    neutral

Current:
    hawkish hold

Then:

stance = HAWKISH
statement_change = MORE_HAWKISH


Another example:

Previous:
    hawkish

Current:
    hawkish

Then:

stance = HAWKISH
statement_change = UNCHANGED


============================================================
SCORING
============================================================

Return a score between -1.0 and +1.0.

Positive = hawkish
Negative = dovish
Near zero = neutral.


Suggested interpretation:

+0.70 to +1.00
    Strong hawkish bias

+0.35 to +0.69
    Clear hawkish bias

+0.20 to +0.34
    Mild hawkish bias

-0.19 to +0.19
    Neutral / balanced

-0.20 to -0.34
    Mild dovish bias

-0.35 to -0.69
    Clear dovish bias

-0.70 to -1.00
    Strong dovish bias


Do not force the score away from zero.

However, do not force HOLD decisions toward zero either.

The score must represent COMMUNICATION BIAS,
not whether the policy rate changed.


============================================================
REQUIRED OUTPUT
============================================================

Return JSON only.

Required fields:

{{
  "stance": "HAWKISH | NEUTRAL | DOVISH",

  "score": 0.0,

  "policy_action":
      "TIGHTENING | HOLD | EASING | OTHER",

  "decision":
      "Concise description of the actual policy decision",

  "target_rate":
      "Target federal funds rate or target range after decision, or null",

  "statement_change":
      "MORE_HAWKISH | MORE_DOVISH | UNCHANGED | NO_COMPARISON",

  "dissent":
      "HAWKISH | DOVISH | MIXED | NONE",

  "policy_drivers": [
      "Policy-bearing evidence only"
  ],

  "key_evidence":
      "Most important policy-bearing sentence or phrase",

  "reason":
      "Short explanation distinguishing actual action from communication bias"
}}


============================================================
CURRENT FOMC STATEMENT
============================================================

{statement_text}


{previous_section}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return json.loads(
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# Calendar
# ============================================================

def normalize_calendar(data):
    """
    Accept:

    [
        {"date": "2026-01-28"}
    ]

    or

    {
        "events": [...]
    }

    or

    {
        "meetings": [...]
    }
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "events",
            "meetings",
            "fomc",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


def get_event_date(event):
    for key in (
        "date",
        "event_date",
        "meeting_date",
        "decision_date",
    ):
        value = event.get(key)

        if value:
            return str(value)[:10]

    return None


# ============================================================
# Main
# ============================================================

def main():
    calendar_raw = load_json(
        CALENDAR_PATH
    )

    calendar = normalize_calendar(
        calendar_raw
    )

    print(
        f"FOMC meetings: {len(calendar)}"
    )

    # ========================================================
    # Load previous-year baseline
    # ========================================================

    print()
    print(
        "Loading previous FOMC baseline: "
        f"{PREVIOUS_FOMC_DATE}"
    )

    previous_statement = fetch_statement(
        PREVIOUS_FOMC_DATE
    )

    if previous_statement:
        previous_statement_text = (
            previous_statement["text"]
        )

        previous_statement_date = (
            PREVIOUS_FOMC_DATE
        )

        print(
            "Baseline statement: OK "
            f"({PREVIOUS_FOMC_DATE})"
        )

    else:
        previous_statement_text = None
        previous_statement_date = None

        print(
            "Baseline statement: NOT FOUND"
        )

    # ========================================================
    # Analyze meetings
    # ========================================================

    results = []

    for event in calendar:
        date = get_event_date(event)

        if not date:
            continue

        print()
        print(f"[{date}]")

        statement = fetch_statement(date)

        if not statement:
            print(
                "  statement: NOT FOUND"
            )

            results.append(
                {
                    "date": date,
                    "event_type": "FOMC",
                    "statement_found": False,
                    "statement_url":
                        get_statement_url(date),
                }
            )

            continue

        print(
            "  statement: OK"
        )

        try:
            signal = analyze_statement(
                statement["text"],
                previous_statement_text,
            )

            result = {
                "date": date,
                "event_type": "FOMC",

                "statement_found": True,

                "statement_url":
                    statement["url"],

                "comparison_date":
                    previous_statement_date,

                **signal,
            }

            results.append(result)

            # =================================================
            # Console
            # =================================================

            if previous_statement_date:
                print(
                    "  comparison: "
                    f"{previous_statement_date}"
                    " -> "
                    f"{date}"
                )

            else:
                print(
                    "  comparison: "
                    "NO PREVIOUS STATEMENT"
                )

            print(
                "  action: "
                f"{signal.get('policy_action')}"
            )

            print(
                "  signal: "
                f"{signal.get('stance')} "
                f"({signal.get('score')})"
            )

            print(
                "  change: "
                f"{signal.get('statement_change')}"
            )

            print(
                "  dissent: "
                f"{signal.get('dissent')}"
            )

            print(
                "  decision: "
                f"{signal.get('decision')}"
            )

            print(
                "  drivers: "
                f"{signal.get('policy_drivers')}"
            )

            print(
                "  reason: "
                f"{signal.get('reason')}"
            )

            # =================================================
            # Current becomes next comparison baseline
            # =================================================

            previous_statement_text = (
                statement["text"]
            )

            previous_statement_date = date

        except Exception as e:
            print(
                f"  LLM ERROR: {e}"
            )

            results.append(
                {
                    "date": date,
                    "event_type": "FOMC",
                    "statement_found": True,
                    "statement_url":
                        statement["url"],
                    "comparison_date":
                        previous_statement_date,
                    "error": str(e),
                }
            )

    # ========================================================
    # Save
    # ========================================================

    save_json(
        OUTPUT_PATH,
        results,
    )

    print()
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Results: {len(results)}"
    )


if __name__ == "__main__":
    main()