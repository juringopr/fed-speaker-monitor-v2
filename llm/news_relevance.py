from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from fed_speaker_monitor_v2.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
)
from fed_speaker_monitor_v2.models import Segment


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class NewsRelevanceResponse(BaseModel):
    """
    News pipeline 1차 LLM gate.

    stance는 판단하지 않는다.

    목적:
    1. 정책 relevance
    2. target speaker 실제 발언 여부
    3. current / historical / commentary 구분
    4. 실제 발언 context 추출
    5. embedding용 normalized text 생성
    """

    policy_relevance_score: int

    policy_relevant: bool

    target_speaker_present: bool

    attribution: Literal[
        "TARGET_SPEAKER",
        "MIXED",
        "OTHER_SPEAKER",
        "UNCLEAR",
    ]

    remark_type: Literal[
        "CURRENT_REMARK",
        "HISTORICAL",
        "COMMENTARY",
        "UNCLEAR",
    ]

    target_text: str

    normalized_target_text: str

    gate_confidence: Literal[
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    reasoning: str


# ============================================================
# SYSTEM PROMPT
# ============================================================

NEWS_RELEVANCE_SYSTEM_PROMPT = """
You are the relevance gate for a Federal Reserve
monetary-policy news analysis pipeline.

Do NOT classify the target speaker as hawkish,
dovish, or neutral.

Your job is to determine whether the supplied news
contains a CURRENT, attributable monetary-policy or
policy-relevant macroeconomic remark from the target
Federal Reserve speaker.


============================================================
1. POLICY RELEVANCE
============================================================

Relevant topics include:

- interest rates
- rate cuts or hikes
- monetary policy
- inflation and price stability
- employment and labor markets
- economic growth and outlook
- financial conditions
- balance-sheet policy
- quantitative tightening or easing
- FOMC policy decisions
- forward guidance

General market reporting is not sufficient by itself.

Examples normally not useful:

- stock, gold, crypto, FX or bond-market moves that
  merely mention the Fed
- company earnings
- event schedules
- "what to watch" articles
- biographies
- political stories that merely mention a Fed official


============================================================
2. TARGET SPEAKER ATTRIBUTION
============================================================

Determine whether the TARGET FEDERAL RESERVE SPEAKER
has an actual attributable policy-relevant view.

Valid evidence includes:

- direct quotation
- clearly attributed paraphrase
- clearly reported policy position
- clearly attributed assessment of inflation,
  employment, growth, financial conditions,
  or monetary policy

Do not treat these as target-speaker evidence:

- another person's opinion about the target speaker
- market interpretation
- speculation about what the speaker may do
- simple name mention
- announcement that the speaker will speak
- another policymaker's statement
- another government official's statement


============================================================
3. ATTRIBUTION
============================================================

TARGET_SPEAKER:
The relevant view is clearly attributable to the
target speaker.

MIXED:
Several people appear, but a relevant target-speaker
statement is clearly identifiable.

OTHER_SPEAKER:
The relevant statement belongs to somebody else.

UNCLEAR:
Available text is insufficient to establish attribution.


============================================================
4. REMARK TYPE
============================================================

Classify the target-speaker material as exactly one:

CURRENT_REMARK:
A current statement, interview, speech, testimony,
reported position, or current policy assessment by
the target speaker.

HISTORICAL:
The article describes a past position, old quotation,
previous Fed tenure, old forecast, or historical
behavior rather than a current remark.

COMMENTARY:
The article discusses, interprets, profiles, predicts,
or evaluates the target speaker without providing a
current attributable policy-relevant remark.

UNCLEAR:
The supplied text is insufficient to determine whether
the statement is current.

Important:
An article published today is NOT automatically a
CURRENT_REMARK.

For example:

"As a Fed governor 15 years ago, Warsh was more
worried about inflation..."

is HISTORICAL, not CURRENT_REMARK.


============================================================
5. TARGET TEXT
============================================================

If a usable target-speaker statement exists, extract
the target speaker's relevant context.

Unlike a short headline fragment, preserve enough
context for later event comparison.

Prefer approximately 1 to 3 sentences.

Preserve when available:

- the target speaker
- the policy view or assessment
- conditional language
- inflation/employment/growth driver
- timing
- current versus historical context

Exclude:

- unrelated market commentary
- other speakers' opinions
- company or asset-price discussion
- unnecessary article background

Do not invent information.

If no usable target-speaker statement exists:

target_text = ""


============================================================
6. NORMALIZED TARGET TEXT
============================================================

Create one concise sentence representing the meaning
of target_text for later semantic event clustering.

The goal is to make differently worded reports of the
same underlying remark more comparable.

Preserve:

- speaker identity
- policy action or assessment
- direction
- conditional language
- main policy driver
- timing or historical status when relevant

Remove:

- publisher wording
- headline style
- redundant adjectives
- unrelated market context

Examples:

Original:
"The Federal Reserve may need to raise interest rates
soon if upcoming data does not show a continued decline
in inflation."

Normalized:
"Collins says rates may need to rise soon if inflation
does not continue declining."

Original:
"US rates need to rise soon absent evidence of ongoing
drop in inflation."

Normalized:
"Collins says rates may need to rise soon if inflation
does not continue declining."

Historical example:

Original:
"As a Fed governor 15 years ago, Warsh was more worried
about inflation than nearly all of his colleagues."

Normalized:
"Historical: Warsh was relatively more concerned about
inflation during his prior Fed tenure 15 years ago."

Do not infer a hawkish, dovish, or neutral label.

If target_text is empty:

normalized_target_text = ""


============================================================
7. POLICY RELEVANCE SCORE
============================================================

Use an integer from 0 to 5.

0:
No meaningful Fed monetary-policy relevance.

1:
Very weak or incidental connection.

2:
Some Fed/macro context but not useful for target-speaker
stance analysis.

3:
Meaningfully related but target-speaker evidence is
limited or indirect.

4:
Clearly relevant with useful target-speaker information.

5:
Strong and explicit target-speaker monetary-policy or
directly policy-relevant economic discussion.


============================================================
8. POLICY_RELEVANT
============================================================

As a general rule:

0-2 -> false
4-5 -> true

For score 3, use judgment.

However, policy_relevant alone does NOT determine whether
the article passes the gate.


============================================================
9. GATE CONFIDENCE
============================================================

This is confidence in the relevance-gate decision.

It is NOT stance confidence.

HIGH:
Evidence clearly supports the decision.

MEDIUM:
Reasonable decision but partial or somewhat ambiguous text.

LOW:
Insufficient or highly ambiguous text.


============================================================
10. IMPORTANT
============================================================

Do NOT determine:

- hawkish
- dovish
- neutral
- stance score
- final policy action classification

Those belong to the later stance stage.

Favor recall when a credible CURRENT target-speaker
policy-relevant remark exists.

Return only the requested structured output.
""".strip()


# ============================================================
# CLIENT
# ============================================================

def _get_client() -> OpenAI:

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# PROMPT
# ============================================================

def build_news_relevance_prompt(
    speaker: str,
    text: str,
) -> str:

    return f"""
Target Federal Reserve speaker:
{speaker}

News text:
{text}

Determine:

1. policy relevance;
2. whether the target speaker has an attributable view;
3. whether that view is CURRENT, HISTORICAL,
   COMMENTARY, or UNCLEAR;
4. target_text with enough context for event comparison;
5. one normalized_target_text sentence for embedding.

Do not classify hawkish/dovish stance.
""".strip()


# ============================================================
# NORMALIZE RESULT
# ============================================================

def _normalize_result(
    result: NewsRelevanceResponse,
) -> NewsRelevanceResponse:

    result.policy_relevance_score = max(
        0,
        min(
            5,
            int(
                result.policy_relevance_score
            ),
        ),
    )

    # 다른 사람 발언 / attribution 불명
    if result.attribution in {
        "OTHER_SPEAKER",
        "UNCLEAR",
    }:
        result.target_speaker_present = False
        result.target_text = ""
        result.normalized_target_text = ""

    # Target attribution인데 실제 text가 없으면
    # usable target statement가 없는 것으로 처리
    if (
        result.attribution
        in {
            "TARGET_SPEAKER",
            "MIXED",
        }
        and not (
            result.target_text
            or ""
        ).strip()
    ):
        result.target_speaker_present = False
        result.normalized_target_text = ""

    # target text가 없으면 normalized도 사용하지 않음
    if not (
        result.target_text
        or ""
    ).strip():
        result.normalized_target_text = ""

    # 최종 gate는 CURRENT remark만 통과
    if (
        not result.target_speaker_present
        or result.remark_type
        != "CURRENT_REMARK"
    ):
        result.policy_relevant = False

    return result


# ============================================================
# ANALYZE ONE
# ============================================================

def analyze_news_relevance(
    segment: Segment,
) -> NewsRelevanceResponse:

    client = _get_client()

    response = client.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content":
                    NEWS_RELEVANCE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    build_news_relevance_prompt(
                        speaker=segment.speaker,
                        text=segment.text,
                    ),
            },
        ],
        text_format=NewsRelevanceResponse,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError(
            "News relevance LLM returned "
            "no parsed output."
        )

    return _normalize_result(
        result
    )


# ============================================================
# PASS / FAIL
# ============================================================

def passes_news_relevance_gate(
    result: NewsRelevanceResponse,
) -> bool:

    return bool(
        result.policy_relevant
        and result.target_speaker_present
        and result.remark_type
        == "CURRENT_REMARK"
    )


# ============================================================
# CACHE
# ============================================================

def _load_cache(
    cache_path: Path | None,
) -> dict[str, dict]:

    if (
        cache_path is None
        or not cache_path.exists()
    ):
        return {}

    try:
        rows = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(
        rows,
        list,
    ):
        return {}

    cache = {}

    for row in rows:

        if not isinstance(
            row,
            dict,
        ):
            continue

        segment_id = str(
            row.get(
                "segment_id",
                "",
            )
            or ""
        )

        if segment_id:
            cache[
                segment_id
            ] = row

    return cache


def _save_cache(
    cache_path: Path | None,
    rows: list[dict],
):

    if cache_path is None:
        return

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# ANALYZE MANY
# ============================================================

def analyze_news_relevance_segments(
    segments: list[Segment],
    cache_path: Path | None = None,
) -> list[dict]:

    cache = _load_cache(
        cache_path
    )

    rows_by_id = dict(
        cache
    )

    reused = 0
    analyzed = 0

    total = len(
        segments
    )

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        segment_id = (
            segment.segment_id
        )

        cached = cache.get(
            segment_id
        )

        # 구버전 cache에는 새 필드가 없으므로 재분석
        if (
            cached is not None
            and "remark_type"
            in cached
            and "normalized_target_text"
            in cached
        ):
            reused += 1

            print(
                f"[RELEVANCE {index}/{total}] "
                f"{segment.speaker} | "
                f"{segment_id} -> CACHE"
            )

            continue

        result = (
            analyze_news_relevance(
                segment
            )
        )

        passed = (
            passes_news_relevance_gate(
                result
            )
        )

        row = {
            "segment_id":
                segment_id,

            "speaker":
                segment.speaker,

            "policy_relevance_score":
                result.policy_relevance_score,

            "policy_relevant":
                result.policy_relevant,

            "target_speaker_present":
                result.target_speaker_present,

            "attribution":
                result.attribution,

            "remark_type":
                result.remark_type,

            "target_text":
                result.target_text,

            "normalized_target_text":
                result.normalized_target_text,

            "gate_confidence":
                result.gate_confidence,

            "reasoning":
                result.reasoning,

            "passed":
                passed,
        }

        rows_by_id[
            segment_id
        ] = row

        analyzed += 1

        print(
            f"[RELEVANCE {index}/{total}] "
            f"{segment.speaker} | "
            f"{segment_id}"
        )

        print(
            "          -> "
            f"{'PASS' if passed else 'DROP'} "
            f"| relevance="
            f"{result.policy_relevance_score}/5 "
            f"| attribution="
            f"{result.attribution} "
            f"| remark="
            f"{result.remark_type} "
            f"| confidence="
            f"{result.gate_confidence}"
        )

        _save_cache(
            cache_path,
            list(
                rows_by_id.values()
            ),
        )

    rows = list(
        rows_by_id.values()
    )

    _save_cache(
        cache_path,
        rows,
    )

    print(
        f"[RELEVANCE CACHE] "
        f"reused={reused} | "
        f"analyzed={analyzed}"
    )

    return rows


# ============================================================
# FILTER SEGMENTS
# ============================================================

def filter_segments_by_relevance(
    segments: list[Segment],
    relevance_rows: list[dict],
) -> list[Segment]:

    passed_ids = {
        str(
            row.get(
                "segment_id",
                "",
            )
        )
        for row in relevance_rows
        if row.get(
            "passed"
        ) is True
    }

    return [
        segment
        for segment in segments
        if segment.segment_id
        in passed_ids
    ]