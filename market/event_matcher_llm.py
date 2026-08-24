import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from fed_speaker_monitor_v2.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MARKET_DIR = (
    BASE_DIR
    / "data"
    / "market"
)

CACHE_PATH = (
    MARKET_DIR
    / "event_llm_cache.json"
)


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class EventMatchResponse(BaseModel):
    """
    두 signal이 같은 실제 Fed policy event인지 판정.
    """

    result: Literal[
        "SAME_EVENT",
        "DIFFERENT_EVENT",
    ]

    confidence: Literal[
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    reason: str


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
# CACHE
# ============================================================

def load_cache():

    if not CACHE_PATH.exists():
        return {}

    try:

        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
    ):

        return {}


def save_cache(cache):

    MARKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CACHE KEY
# ============================================================

def get_signal_id(row):
    """
    segment_id를 우선 사용.

    없으면 최소한의 fallback key 생성.
    """

    segment_id = row.get(
        "segment_id"
    )

    if segment_id:
        return str(segment_id)

    return "|".join(
        [
            str(
                row.get("source_type")
                or ""
            ),
            str(
                row.get("speaker")
                or ""
            ),
            str(
                row.get("date")
                or ""
            ),
            str(
                row.get("title")
                or ""
            ),
            str(
                row.get(
                    "policy_bearing_phrase"
                )
                or ""
            ),
        ]
    )


def make_cache_key(
    signal_a,
    signal_b,
):
    """
    A-B / B-A가 동일 key가 되도록 정렬.
    """

    id_a = get_signal_id(
        signal_a
    )

    id_b = get_signal_id(
        signal_b
    )

    return "||".join(
        sorted(
            [
                id_a,
                id_b,
            ]
        )
    )


# ============================================================
# TEXT
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def build_signal_text(
    label,
    row,
):

    return f"""
SIGNAL {label}

Source Type:
{clean_text(row.get("source_type"))}

Speaker:
{clean_text(row.get("speaker"))}

Date:
{clean_text(row.get("date"))}

Stance:
{clean_text(row.get("stance"))}

Score:
{clean_text(
    row.get("weighted_score")
    if row.get("source_type") == "NEWS"
    else row.get("score")
)}

Policy Action:
{clean_text(row.get("policy_action"))}

Stance Driver:
{clean_text(row.get("stance_driver"))}

Policy-bearing Phrase:
{clean_text(row.get("policy_bearing_phrase"))}

Evidence:
{clean_text(row.get("evidence"))}

Title:
{clean_text(row.get("title"))}

URL:
{clean_text(row.get("url"))}
""".strip()


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are evaluating whether two Federal Reserve policy signals
refer to the same underlying real-world policy event.

The purpose is event deduplication.

SAME_EVENT means that both signals refer to the same underlying
speech, interview, testimony, FOMC vote, official statement,
press conference, or the same specific policy remark that was
reported by multiple sources.

DIFFERENT_EVENT means they refer to separate real-world remarks,
speeches, interviews, votes, statements, or policy events.

Important rules:

1. Similar policy views alone do NOT make two signals the same event.

2. Two articles can use very different wording while still referring
   to the same underlying remark.

3. Two signals from the same speaker on the same day can still be
   DIFFERENT_EVENT if they concern distinct remarks or policy actions.

4. An official statement and a news article can be SAME_EVENT when
   the news article is clearly reporting that official statement,
   vote, speech, or remark.

5. Do not classify two signals as SAME_EVENT merely because both are
   hawkish or both are dovish.

6. Focus on whether they describe the same underlying real-world
   occurrence, not whether their language is textually similar.

7. If the available evidence is insufficient to establish that they
   are the same event, prefer DIFFERENT_EVENT with LOW confidence.

Return only the structured result.
""".strip()


def build_event_match_prompt(
    signal_a,
    signal_b,
):

    signal_a_text = build_signal_text(
        "A",
        signal_a,
    )

    signal_b_text = build_signal_text(
        "B",
        signal_b,
    )

    return f"""
Determine whether SIGNAL A and SIGNAL B refer to the same
underlying Federal Reserve policy event.

{signal_a_text}


{signal_b_text}


Classify as:

SAME_EVENT
or
DIFFERENT_EVENT

Also provide confidence and a short reason.
""".strip()


# ============================================================
# PRECHECK
# ============================================================

def precheck(
    signal_a,
    signal_b,
):
    """
    명백히 다른 경우에는 LLM 호출하지 않는다.
    """

    # --------------------------------------------------------
    # Speaker 다름
    # --------------------------------------------------------

    if (
        signal_a.get("speaker")
        != signal_b.get("speaker")
    ):

        return {
            "result":
                "DIFFERENT_EVENT",

            "confidence":
                "HIGH",

            "reason":
                "Different speakers.",

            "source":
                "RULE",
        }


    # --------------------------------------------------------
    # Date 다름
    # --------------------------------------------------------

    if (
        signal_a.get("date")
        != signal_b.get("date")
    ):

        return {
            "result":
                "DIFFERENT_EVENT",

            "confidence":
                "HIGH",

            "reason":
                "Different event dates.",

            "source":
                "RULE",
        }


    # --------------------------------------------------------
    # Stance 다름
    # --------------------------------------------------------

    if (
        signal_a.get("stance")
        != signal_b.get("stance")
    ):

        return {
            "result":
                "DIFFERENT_EVENT",

            "confidence":
                "HIGH",

            "reason":
                "Different policy stance.",

            "source":
                "RULE",
        }


    return None


# ============================================================
# LLM MATCH
# ============================================================

def match_policy_events(
    signal_a,
    signal_b,
    use_cache=True,
):
    """
    두 signal이 같은 실제 Fed policy event인지 판정.

    1. 명백한 차이는 rule로 제거
    2. cache 확인
    3. 필요한 경우에만 LLM 호출
    """

    # --------------------------------------------------------
    # 1. Rule precheck
    # --------------------------------------------------------

    rule_result = precheck(
        signal_a,
        signal_b,
    )

    if rule_result:
        return rule_result


    # --------------------------------------------------------
    # 2. Cache
    # --------------------------------------------------------

    cache_key = make_cache_key(
        signal_a,
        signal_b,
    )

    cache = (
        load_cache()
        if use_cache
        else {}
    )


    if (
        use_cache
        and cache_key in cache
    ):

        cached = dict(
            cache[cache_key]
        )

        cached["source"] = (
            "CACHE"
        )

        return cached


    # --------------------------------------------------------
    # 3. LLM
    # --------------------------------------------------------

    client = _get_client()

    prompt = (
        build_event_match_prompt(
            signal_a,
            signal_b,
        )
    )


    response = client.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    prompt,
            },
        ],
        text_format=EventMatchResponse,
    )


    parsed = response.output_parsed


    if parsed is None:
        raise RuntimeError(
            "Event matcher returned no parsed output."
        )


    result = {
        "result":
            parsed.result,

        "confidence":
            parsed.confidence,

        "reason":
            parsed.reason,

        "source":
            "LLM",
    }


    # --------------------------------------------------------
    # 4. Save cache
    # --------------------------------------------------------

    if use_cache:

        cache[cache_key] = {
            "result":
                parsed.result,

            "confidence":
                parsed.confidence,

            "reason":
                parsed.reason,
        }

        save_cache(
            cache
        )


    return result


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    signals_path = (
        MARKET_DIR
        / "strong_signals.json"
    )


    with open(
        signals_path,
        "r",
        encoding="utf-8",
    ) as f:

        signals = json.load(f)


    # --------------------------------------------------------
    # 테스트할 후보 찾기
    #
    # same speaker + date + stance인 첫 pair
    # --------------------------------------------------------

    test_pair = None


    for i, signal_a in enumerate(
        signals
    ):

        for signal_b in signals[
            i + 1:
        ]:

            if (
                signal_a.get("speaker")
                == signal_b.get("speaker")
                and signal_a.get("date")
                == signal_b.get("date")
                and signal_a.get("stance")
                == signal_b.get("stance")
            ):

                test_pair = (
                    signal_a,
                    signal_b,
                )

                break

        if test_pair:
            break


    if not test_pair:

        print(
            "No candidate pair found."
        )

    else:

        signal_a, signal_b = (
            test_pair
        )

        print("=" * 80)
        print("EVENT MATCH TEST")
        print("=" * 80)

        print(
            "Speaker:",
            signal_a.get("speaker"),
        )

        print(
            "Date:",
            signal_a.get("date"),
        )

        print()

        print(
            "A:",
            signal_a.get(
                "policy_bearing_phrase"
            )
            or signal_a.get(
                "evidence"
            )
            or signal_a.get(
                "title"
            ),
        )

        print()

        print(
            "B:",
            signal_b.get(
                "policy_bearing_phrase"
            )
            or signal_b.get(
                "evidence"
            )
            or signal_b.get(
                "title"
            ),
        )

        print()

        result = match_policy_events(
            signal_a,
            signal_b,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        print("=" * 80)