from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

from openai import OpenAI

from fed_speaker_monitor_v2.llm.macro_context import build_macro_context


# ============================================================
# 설정
# ============================================================

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini",
)


# ============================================================
# 결과 모델
#
# 이 결과는 stance가 아니다.
# 발언 당시 경제환경을 설명하기 위한 background context다.
# ============================================================

@dataclass
class MacroBackground:
    economic_condition: str
    recent_direction: str
    key_evidence: str
    mixed_signals: str
    confidence: str


# ============================================================
# 허용값
# ============================================================

VALID_DIRECTIONS = {
    "STRENGTHENING",
    "WEAKENING",
    "ACCELERATING",
    "MODERATING",
    "STABLE",
    "MIXED",
    "UNCLEAR",
}

VALID_CONFIDENCE = {
    "HIGH",
    "MEDIUM",
    "LOW",
}


# ============================================================
# Macro Background Prompt
# ============================================================

MACRO_BACKGROUND_SYSTEM_PROMPT = """
You summarize the macroeconomic environment that was available
at the time of a Federal Reserve statement.

Your task is to provide BACKGROUND CONTEXT ONLY.

You are NOT classifying the Federal Reserve speaker.

You are NOT determining whether the speaker is hawkish or dovish.

You are NOT recommending monetary policy.

You are NOT determining whether interest rates should rise or fall.

Your task is only to summarize what the available economic data
indicated about the relevant part of the economy at that time.

============================================================
CORE PRINCIPLE
============================================================

The macroeconomic data is background information.

Use it to answer questions such as:

- Was inflation accelerating, moderating, stable, or mixed?
- Was the labor market strengthening, weakening, stable, or mixed?
- Was economic growth strengthening, weakening, stable, or mixed?
- Were financial conditions tightening, easing, stable, or mixed?

Do not convert these observations into monetary-policy conclusions.

For example:

ACCEPTABLE:

Payroll growth weakened and wage growth undershot expectations,
while the unemployment rate remained relatively low.

NOT ACCEPTABLE:

The weaker labor market supports rate cuts.

ACCEPTABLE:

Inflation remained elevated but recent price data showed
some moderation.

NOT ACCEPTABLE:

Inflation remains high, so the Fed should maintain
restrictive rates.

============================================================
DATA INTERPRETATION
============================================================

The supplied data may contain:

actual
consensus
forecast
previous
vs_consensus
vs_previous

Interpret ABOVE, BELOW, and INLINE according to the meaning
of the specific economic indicator.

Do NOT assume:

ABOVE = strong
BELOW = weak

and never assume:

ABOVE = hawkish
BELOW = dovish.

Examples:

Payroll employment BELOW consensus
may indicate weaker hiring.

Unemployment rate BELOW consensus
may indicate stronger labor conditions.

Average hourly earnings BELOW consensus
may indicate softer wage pressure.

Initial jobless claims ABOVE consensus
may indicate weaker labor conditions.

CPI ABOVE consensus
may indicate stronger inflation pressure.

GDP growth BELOW consensus
may indicate weaker economic growth.

============================================================
HIGH-FREQUENCY DATA
============================================================

Distinguish between major lower-frequency indicators and
high-frequency supplementary indicators.

Major monthly or quarterly indicators should normally provide
the primary evidence for the broad macroeconomic condition.

Examples include:

- nonfarm payrolls,
- unemployment,
- average hourly earnings,
- JOLTS,
- CPI,
- PCE,
- GDP,
- retail sales,
- industrial production,
- major PMI measures.

Weekly or other high-frequency indicators should be used as
supplementary evidence about the most recent direction.

Examples include:

- initial jobless claims,
- weekly or four-week-average employment measures,
- other frequently released indicators.

Do NOT give a high-frequency indicator greater importance merely
because multiple observations of that indicator appear in the data.

Repeated weekly observations are multiple observations of the same
economic signal, not multiple independent major indicators.

For high-frequency indicators, focus on the direction across
consecutive observations rather than treating each observation
as separate evidence of equal importance.

Use high-frequency indicators primarily to:

- confirm the direction suggested by major indicators,
- weaken confidence in that direction,
- identify a possible recent change,
- or qualify the broader assessment.

If major monthly indicators and high-frequency indicators disagree,
describe the disagreement explicitly.

Do not mechanically ignore high-frequency data.
Recent high-frequency data can provide useful information about
conditions developing after the latest major monthly release.

============================================================
INDICATOR IMPORTANCE AND FREQUENCY
============================================================

When assessing the macroeconomic background:

- Monthly major indicators such as nonfarm payrolls,
  unemployment, wages, JOLTS, CPI, PCE, GDP and retail sales
  should be used to assess the broader macroeconomic condition.

- Weekly or high-frequency indicators such as initial jobless
  claims and weekly ADP measures should be used as supplementary
  evidence about the most recent direction of the economy.

- Do not let repeated weekly observations dominate the assessment
  merely because they appear more frequently in the data.

- For weekly indicators, focus primarily on the direction across
  multiple consecutive observations rather than any single release.

- Use high-frequency indicators to confirm, weaken, or qualify
  the picture from major monthly indicators.

- If major monthly indicators and high-frequency indicators disagree,
  explicitly describe the disagreement as a mixed signal.

- The number of repeated observations must not be interpreted as
  greater evidentiary weight.

============================================================
RECENT DIRECTION
============================================================

Determine the recent direction of the relevant economic phenomenon.

Use exactly one:

STRENGTHENING
WEAKENING
ACCELERATING
MODERATING
STABLE
MIXED
UNCLEAR

Choose the label according to the economic phenomenon.

Examples:

Labor demand becoming stronger:
STRENGTHENING

Labor demand becoming softer:
WEAKENING

Inflation pressure increasing:
ACCELERATING

Inflation pressure easing:
MODERATING

Growth showing conflicting indicators:
MIXED

Insufficient information:
UNCLEAR

Do not interpret the direction as a monetary-policy stance.

============================================================
ECONOMIC CONDITION
============================================================

Summarize the current economic condition represented by the
available data.

Keep the statement factual and concise.

Good examples:

"Labor-market momentum has weakened recently."

"Inflation remains elevated but recent price pressures have moderated."

"Growth remains positive, although recent activity indicators are mixed."

Do not mention:

hawkish
dovish
tightening
easing
rate hikes
rate cuts
policy preference

unless those words are literally part of an indicator name,
which is unlikely.

============================================================
KEY EVIDENCE
============================================================

Identify the most informative economic indicators supporting
the assessment.

Prefer:

1. major indicators,
2. recent releases,
3. indicators with actual and consensus values,
4. indicators showing meaningful changes from previous readings.

High-frequency indicators may be included when they provide useful
information about the most recent direction, but repeated releases
of the same weekly indicator should be summarized as a trend rather
than counted as separate major pieces of evidence.

Do not simply repeat every supplied indicator.

Summarize the strongest evidence.

============================================================
MIXED SIGNALS
============================================================

Identify evidence that does not fit the dominant direction.

For example:

Payroll growth may weaken while the unemployment rate remains low.

Headline inflation may moderate while wage or producer-price
pressures remain firm.

High-frequency indicators may also be used here when their recent
direction conflicts with the broader picture from major indicators.

If there are no meaningful conflicting signals, return:

"None"

Do not invent conflicting evidence.

============================================================
CONFIDENCE
============================================================

Use exactly one:

HIGH
MEDIUM
LOW

HIGH:
Several important and recent indicators point in a consistent direction.

MEDIUM:
The broad direction is identifiable but some indicators conflict
or important data is missing.

LOW:
The available data is sparse, old, ambiguous, or strongly conflicting.

The presence of many repeated weekly observations alone should not
increase confidence.

Confidence should reflect the breadth, importance, recency, and
consistency of the evidence rather than the raw number of data rows.

Confidence refers only to the macroeconomic background assessment.

It does NOT refer to confidence in a hawkish or dovish classification.

============================================================
IMPORTANT RESTRICTION
============================================================

Never produce:

- HAWKISH
- DOVISH
- MORE_RESTRICTIVE
- LESS_RESTRICTIVE
- policy recommendation
- preferred interest-rate direction

The final stance model will make its own policy classification.

Your output is background context only.

============================================================
OUTPUT
============================================================

Return JSON only.

Use exactly this schema:

{
  "economic_condition": "...",
  "recent_direction": "MIXED",
  "key_evidence": "...",
  "mixed_signals": "...",
  "confidence": "MEDIUM"
}

Do not include markdown.

Do not include additional fields.
"""


# ============================================================
# User Prompt 생성
# ============================================================

def _build_user_prompt(
    driver: str,
    as_of: str | datetime,
    macro_context: str,
) -> str:

    return f"""
MACROECONOMIC AREA:

{driver}


AS OF:

{as_of}


MACROECONOMIC DATA AVAILABLE AT THAT TIME:

{macro_context}


Summarize the macroeconomic background represented by these data.

Focus on:

1. the current economic condition,
2. the recent direction,
3. the strongest supporting indicators,
4. meaningful conflicting signals,
5. confidence in the macro assessment.

This is background context only.

Do not classify any Federal Reserve speaker.

Do not infer hawkish or dovish policy.

Do not recommend tighter or easier monetary policy.

Return JSON only.
""".strip()


# ============================================================
# JSON 파싱
# ============================================================

def _parse_result(
    content: str,
) -> MacroBackground:

    data = json.loads(content)

    recent_direction = str(
        data.get(
            "recent_direction",
            "UNCLEAR",
        )
    ).upper()

    confidence = str(
        data.get(
            "confidence",
            "LOW",
        )
    ).upper()

    if recent_direction not in VALID_DIRECTIONS:
        recent_direction = "UNCLEAR"

    if confidence not in VALID_CONFIDENCE:
        confidence = "LOW"

    return MacroBackground(
        economic_condition=str(
            data.get(
                "economic_condition",
                "",
            )
        ).strip(),

        recent_direction=recent_direction,

        key_evidence=str(
            data.get(
                "key_evidence",
                "",
            )
        ).strip(),

        mixed_signals=str(
            data.get(
                "mixed_signals",
                "",
            )
        ).strip(),

        confidence=confidence,
    )


# ============================================================
# Macro Background 분석
# ============================================================

def analyze_macro_background(
    driver: str,
    as_of: str | datetime,
    lookback_days: int = 90,
    max_macro_items: int = 17,
    model: str = MODEL,
) -> MacroBackground:

    driver = (
        driver.upper().strip()
        if driver
        else "OTHER"
    )

    # --------------------------------------------------------
    # 발언 당시 이용 가능했던 Macro Context
    # --------------------------------------------------------

    macro_context = build_macro_context(
        as_of=as_of,
        driver=driver,
        lookback_days=lookback_days,
        max_items=max_macro_items,
    )

    # --------------------------------------------------------
    # 관련 macro context가 없는 경우
    #
    # POLICY_FRAMEWORK / OTHER 등은 macro_context.py에서
    # 관련 지표를 반환하지 않도록 되어 있다.
    # --------------------------------------------------------

    if (
        not macro_context
        or macro_context.strip()
        == "No relevant macroeconomic context available."
    ):

        return MacroBackground(
            economic_condition=(
                "No relevant macroeconomic context available."
            ),
            recent_direction="UNCLEAR",
            key_evidence="None",
            mixed_signals="None",
            confidence="LOW",
        )

    user_prompt = _build_user_prompt(
        driver=driver,
        as_of=as_of,
        macro_context=macro_context,
    )

    # --------------------------------------------------------
    # LLM 호출
    #
    # temperature는 지정하지 않는다.
    # 일부 최신 모델은 기본값만 지원한다.
    # --------------------------------------------------------

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": MACRO_BACKGROUND_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        return MacroBackground(
            economic_condition="",
            recent_direction="UNCLEAR",
            key_evidence="",
            mixed_signals="",
            confidence="LOW",
        )

    return _parse_result(content)


# ============================================================
# Stance Prompt에 넣기 위한 문자열
# ============================================================

def format_macro_background(
    background: MacroBackground,
) -> str:

    return (
        f"Economic condition: "
        f"{background.economic_condition}\n"
        f"Recent direction: "
        f"{background.recent_direction}\n"
        f"Key evidence: "
        f"{background.key_evidence}\n"
        f"Mixed signals: "
        f"{background.mixed_signals}\n"
        f"Background confidence: "
        f"{background.confidence}"
    )


# ============================================================
# Stance.py에서 사용할 최종 함수
# ============================================================

def build_macro_background(
    driver: str,
    as_of: str | datetime,
    lookback_days: int = 90,
    max_macro_items: int = 12,
    model: str = MODEL,
) -> str:

    background = analyze_macro_background(
        driver=driver,
        as_of=as_of,
        lookback_days=lookback_days,
        max_macro_items=max_macro_items,
        model=model,
    )

    return format_macro_background(
        background
    )


# ============================================================
# 단독 테스트
# ============================================================

if __name__ == "__main__":

    test_date = "2026-08-20 23:59"

    tests = [
        "INFLATION",
        "LABOR",
        "GROWTH",
        "FINANCIAL_CONDITIONS",
        "POLICY_FRAMEWORK",
    ]

    for driver in tests:

        print("\n" + "=" * 100)
        print("DRIVER:", driver)
        print("=" * 100)

        result = analyze_macro_background(
            driver=driver,
            as_of=test_date,
        )

        print(
            "economic_condition:",
            result.economic_condition,
        )

        print(
            "recent_direction:",
            result.recent_direction,
        )

        print(
            "key_evidence:",
            result.key_evidence,
        )

        print(
            "mixed_signals:",
            result.mixed_signals,
        )

        print(
            "confidence:",
            result.confidence,
        )