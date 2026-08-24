from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from io import BytesIO
from pypdf import PdfReader
from urllib.parse import urljoin


BASE_DIR = Path(__file__).resolve().parents[1]

CALENDAR_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_calendar.json"
)

POLICY_SIGNAL_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_policy_signals.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_chair_communication.json"
)

FED_BASE_URL = "https://www.federalreserve.gov"

FOMC_CALENDAR_URL = (
    "https://www.federalreserve.gov/"
    "monetarypolicy/fomccalendars.htm"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


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
# Calendar normalization
# ============================================================

def normalize_calendar(data):
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
# Statement signal lookup
# ============================================================

def load_policy_signals():
    if not POLICY_SIGNAL_PATH.exists():
        return {}

    data = load_json(
        POLICY_SIGNAL_PATH
    )

    result = {}

    for row in data:
        date = row.get("date")

        if date:
            result[date] = row

    return result


# ============================================================
# Press conference transcript discovery
# ============================================================

def find_press_conference_url(date: str):
    """
    Search the Federal Reserve FOMC calendar page
    for a press conference transcript associated
    with the FOMC date.

    Preference:
    1. Press conference transcript
    2. Press conference page
    """

    try:
        response = requests.get(
            FOMC_CALENDAR_URL,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

    except requests.RequestException:
        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    yyyymmdd = date.replace("-", "")

    candidates = []

    for link in soup.find_all(
        "a",
        href=True,
    ):
        href = link.get("href", "")
        text = link.get_text(
            " ",
            strip=True,
        )

        href_lower = href.lower()
        text_lower = text.lower()

        # Date should normally appear in the Fed URL.
        if yyyymmdd not in href:
            continue

        if (
            "press conference" in text_lower
            or "pressconference" in href_lower
            or "fomcpresconf" in href_lower
        ):
            candidates.append(
                (
                    text,
                    href,
                )
            )

    if not candidates:
        return None

    # Prefer transcript / HTML page.
    candidates.sort(
        key=lambda x: (
            "transcript" not in x[0].lower(),
            ".pdf" in x[1].lower(),
        )
    )

    href = candidates[0][1]

    if href.startswith("http"):
        return href

    return FED_BASE_URL + href


# ============================================================
# Transcript fetch
# ============================================================

def fetch_press_conference(date: str):
    """
    1. Find the official FOMC press conference page.
    2. Open the page.
    3. Find the official Press Conference Transcript PDF.
    4. Extract the PDF text.
    """

    page_url = find_press_conference_url(date)

    if not page_url:
        return None

    try:
        response = requests.get(
            page_url,
            headers=HEADERS,
            timeout=30,
        )

        if response.status_code != 200:
            return None

    except requests.RequestException:
        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    transcript_url = None

    # Find the actual Press Conference Transcript PDF
    for link in soup.find_all("a", href=True):
        text = link.get_text(
            " ",
            strip=True,
        ).lower()

        href = link.get("href", "")

        if (
            "press conference transcript" in text
            and ".pdf" in href.lower()
        ):
            transcript_url = urljoin(
                page_url,
                href,
            )
            break

    if not transcript_url:
        return None

    try:
        pdf_response = requests.get(
            transcript_url,
            headers=HEADERS,
            timeout=30,
        )

        if pdf_response.status_code != 200:
            return None

    except requests.RequestException:
        return None

    try:
        reader = PdfReader(
            BytesIO(pdf_response.content)
        )

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        transcript_text = "\n".join(pages)

        transcript_text = re.sub(
            r"\n{3,}",
            "\n\n",
            transcript_text,
        )

    except Exception:
        return None

    if len(transcript_text) < 1000:
        return None

    return {
        "url": transcript_url,
        "page_url": page_url,
        "text": transcript_text,
        "format": "PDF",
    }


# ============================================================
# LLM
# ============================================================

def analyze_press_conference(
    date: str,
    transcript: str,
    policy_signal: dict | None,
):
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        )
    )

    if policy_signal:
        statement_context = json.dumps(
            {
                "stance": policy_signal.get(
                    "stance"
                ),
                "score": policy_signal.get(
                    "score"
                ),
                "policy_action": (
                    policy_signal.get(
                        "policy_action"
                    )
                ),
                "decision": policy_signal.get(
                    "decision"
                ),
                "statement_change": (
                    policy_signal.get(
                        "statement_change"
                    )
                    or policy_signal.get(
                        "change"
                    )
                ),
                "dissent": policy_signal.get(
                    "dissent"
                ),
                "reason": policy_signal.get(
                    "reason"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    else:
        statement_context = (
            "No statement analysis available."
        )

    prompt = f"""
You are analyzing the Federal Reserve Chair's
official FOMC press conference.

The purpose is NOT to determine the Chair's
general personal hawkish or dovish ideology.

The purpose is to determine whether the Chair's
communication at THIS press conference adds a
hawkish, neutral, or dovish policy signal to the
official FOMC decision.

FOMC DATE:
{date}

OFFICIAL STATEMENT ANALYSIS:
{statement_context}


CLASSIFICATION
==============

Return JSON only.

Required fields:

press_conference_signal:
- HAWKISH
- NEUTRAL
- DOVISH

press_conference_score:
- number between -1.0 and +1.0
- positive = hawkish
- negative = dovish
- 0 = neutral

communication_change:
- MORE_HAWKISH
- UNCHANGED
- MORE_DOVISH
- NO_COMPARISON

policy_bias:
- TIGHTENING_BIAS
- BALANCED
- EASING_BIAS

key_evidence:
An array containing the most important
policy-bearing phrases from the Chair.

reason:
A concise explanation of the classification.


IMPORTANT RULES
===============

1. Separate economic description from policy signal.

Statements such as:
"inflation remains elevated"
"the labor market has softened"
"uncertainty remains high"

are NOT by themselves hawkish or dovish.


2. Focus on policy-bearing communication.

Examples include statements about:

- likelihood of future rate hikes
- likelihood of future rate cuts
- conditions required before changing rates
- whether policy is sufficiently restrictive
- whether additional tightening may be needed
- whether easing may soon be appropriate
- willingness to tolerate inflation risks
- willingness to tolerate employment risks
- timing of future policy adjustments


3. Compare the press conference with the
official FOMC statement.

communication_change asks:

Did the Chair make the overall policy message
more hawkish or more dovish than the written
statement?

Example:

Statement:
HOLD / NEUTRAL

Chair:
"Further tightening may be appropriate."

Then:

press_conference_signal = HAWKISH
communication_change = MORE_HAWKISH


4. A HOLD decision can still have a hawkish
or dovish communication signal.

Do not automatically classify HOLD as NEUTRAL.


5. Do NOT classify the communication based only
on dissenting votes.

Dissent is useful context, but the target is
the Chair's communication.


6. Do NOT infer a directional signal merely
because the Chair refuses to rule out future
moves.

Standard optionality such as:

"We will assess incoming data."

or

"We are prepared to adjust policy as appropriate."

should normally be NEUTRAL unless accompanied
by meaningful directional guidance.


7. Prefer explicit policy-bearing language over
journalistic interpretation or market reaction.

Use only the official press conference text
provided below.


OFFICIAL PRESS CONFERENCE:
==========================

{transcript}
"""

    response = (
        client.chat.completions.create(
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
    )

    return json.loads(
        response
        .choices[0]
        .message.content
    )


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

    policy_signals = (
        load_policy_signals()
    )

    print(
        f"FOMC meetings: {len(calendar)}"
    )

    results = []

    for event in calendar:
        date = get_event_date(event)

        if not date:
            continue

        print()
        print(f"[{date}]")

        press = fetch_press_conference(
            date
        )

        if not press:
            print(
                "  press conference: NOT FOUND"
            )

            results.append(
                {
                    "date": date,
                    "event_type": "FOMC",
                    "press_conference_found": False,
                }
            )

            continue

        print(
            "  press conference: OK"
        )
        print(
            f"  format: {press['format']}"
        )

        if not press.get("text"):
            print(
                "  transcript text: NOT AVAILABLE"
            )

            results.append(
                {
                    "date": date,
                    "event_type": "FOMC",
                    "press_conference_found": True,
                    "press_conference_url": (
                        press["url"]
                    ),
                    "format": press["format"],
                    "analysis_available": False,
                }
            )

            continue

        policy_signal = (
            policy_signals.get(date)
        )

        try:
            signal = analyze_press_conference(
                date=date,
                transcript=press["text"],
                policy_signal=policy_signal,
            )

            result = {
                "date": date,
                "event_type": "FOMC",
                "press_conference_found": True,
                "press_conference_url": (
                    press["url"]
                ),
                "format": press["format"],
                "analysis_available": True,
                **signal,
            }

            results.append(result)

            print(
                "  signal: "
                f"{signal.get('press_conference_signal')} "
                f"({signal.get('press_conference_score')})"
            )

            print(
                "  vs statement: "
                f"{signal.get('communication_change')}"
            )

            print(
                "  bias: "
                f"{signal.get('policy_bias')}"
            )

            print(
                "  reason: "
                f"{signal.get('reason')}"
            )

        except Exception as e:
            print(
                f"  LLM ERROR: {e}"
            )

            results.append(
                {
                    "date": date,
                    "event_type": "FOMC",
                    "press_conference_found": True,
                    "press_conference_url": (
                        press["url"]
                    ),
                    "format": press["format"],
                    "analysis_available": False,
                    "error": str(e),
                }
            )

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