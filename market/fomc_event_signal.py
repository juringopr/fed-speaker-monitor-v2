from __future__ import annotations

import json
import re
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

POLICY_SIGNAL_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_policy_signals.json"
)

CHAIR_SIGNAL_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_chair_communication.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "market"
    / "fomc_event_signals.json"
)


# ============================================================
# Settings
# ============================================================

DEFAULT_STATEMENT_WEIGHT = 0.8
DEFAULT_CHAIR_WEIGHT = 0.2

INCREMENTAL_STATEMENT_WEIGHT = 0.5
INCREMENTAL_CHAIR_WEIGHT = 0.5

HAWKISH_THRESHOLD = 0.20
DOVISH_THRESHOLD = -0.20

INCREMENTAL_CHANGES = {
    "MORE_HAWKISH",
    "MORE_DOVISH",
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
# Helpers
# ============================================================

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_rate_text(text: str | None):
    if not text:
        return ""

    return (
        str(text)
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )


def extract_target_range(row: dict):
    """
    Extract target range from target_rate first.

    Fallback:
        parse the decision text.

    Examples:
        3.5 to 3.75 percent
        3-1/2 to 3-3/4 percent
        3.50-3.75%
    """

    texts = [
        row.get("target_rate"),
        row.get("decision"),
    ]

    for raw_text in texts:
        text = normalize_rate_text(raw_text)

        if not text:
            continue

        # --------------------------------------------
        # 3-1/2 to 3-3/4
        # --------------------------------------------

        mixed = re.search(
            r"(\d+)\s*-\s*1/2"
            r"\s*(?:to|-)\s*"
            r"(\d+)\s*-\s*3/4",
            text,
            re.IGNORECASE,
        )

        if mixed:
            lower = float(mixed.group(1)) + 0.5
            upper = float(mixed.group(2)) + 0.75

            return lower, upper

        # --------------------------------------------
        # 3.5 to 3.75
        # 3.50 - 3.75
        # --------------------------------------------

        decimal = re.search(
            r"(\d+(?:\.\d+)?)"
            r"\s*(?:to|-)\s*"
            r"(\d+(?:\.\d+)?)"
            r"\s*(?:percent|%)?",
            text,
            re.IGNORECASE,
        )

        if decimal:
            lower = float(decimal.group(1))
            upper = float(decimal.group(2))

            return lower, upper

    return None


def get_midpoint(target_range):
    if not target_range:
        return None

    lower, upper = target_range

    return (lower + upper) / 2


def classify_action(rate_change_bp):
    if rate_change_bp is None:
        return "UNKNOWN"

    if rate_change_bp > 0:
        return "HIKE"

    if rate_change_bp < 0:
        return "CUT"

    return "HOLD"


def classify_stance(score):
    if score is None:
        return "UNKNOWN"

    if score >= HAWKISH_THRESHOLD:
        return "HAWKISH"

    if score <= DOVISH_THRESHOLD:
        return "DOVISH"

    return "NEUTRAL"


# ============================================================
# Labels
# ============================================================

def format_event_label(
    stance: str,
    action: str,
    rate_change_bp,
):
    """
    English event label.

    Examples:
        HAWKISH HOLD
        DOVISH HOLD
        DOVISH CUT -25BP
        HAWKISH HIKE +25BP
    """

    if action == "HOLD":
        return f"{stance} HOLD"

    if (
        action == "CUT"
        and rate_change_bp is not None
    ):
        bp = abs(int(round(rate_change_bp)))

        return f"{stance} CUT -{bp}BP"

    if (
        action == "HIKE"
        and rate_change_bp is not None
    ):
        bp = abs(int(round(rate_change_bp)))

        return f"{stance} HIKE +{bp}BP"

    return f"{stance} {action}"


def format_event_label_ko(
    stance: str,
    action: str,
    rate_change_bp,
):
    """
    Korean event label.

    Examples:
        매파적 동결
        비둘기적 동결
        비둘기적 25bp 인하
        매파적 25bp 인상
    """

    stance_map = {
        "HAWKISH": "매파적",
        "NEUTRAL": "중립적",
        "DOVISH": "비둘기적",
        "UNKNOWN": "불명확",
    }

    action_map = {
        "HOLD": "동결",
        "HIKE": "인상",
        "CUT": "인하",
        "UNKNOWN": "결정 불명확",
    }

    stance_ko = stance_map.get(
        stance,
        stance,
    )

    action_ko = action_map.get(
        action,
        action,
    )

    if (
        action in {"HIKE", "CUT"}
        and rate_change_bp is not None
    ):
        bp = abs(int(round(rate_change_bp)))

        return (
            f"{stance_ko} "
            f"{bp}bp "
            f"{action_ko}"
        )

    return f"{stance_ko} {action_ko}"


def format_shift_label_ko(
    change: str | None,
):
    """
    Translate relative communication change.
    """

    shift_map = {
        "MORE_HAWKISH":
            "직전 회의 대비 매파적 변화",

        "MORE_DOVISH":
            "직전 회의 대비 비둘기적 변화",

        "UNCHANGED":
            "직전 회의 대비 변화 없음",

        "NO_COMPARISON":
            "비교 불가",
    }

    if not change:
        return None

    return shift_map.get(
        change,
        change,
    )


# ============================================================
# Chair lookup
# ============================================================

def load_chair_lookup():
    if not CHAIR_SIGNAL_PATH.exists():
        return {}

    data = load_json(
        CHAIR_SIGNAL_PATH
    )

    lookup = {}

    for row in data:
        date = row.get("date")

        if date:
            lookup[date] = row

    return lookup


# ============================================================
# Weighting
# ============================================================

def calculate_communication_signal(
    statement_score,
    chair_row,
):
    """
    Statement is the primary FOMC communication signal.

    Default:
        Statement 80%
        Chair     20%

    If Chair communication adds meaningful
    directional information:
        Statement 50%
        Chair     50%

    If Chair analysis is unavailable:
        Statement 100%
    """

    statement_score = (
        to_float(statement_score)
        if statement_score is not None
        else 0.0
    )

    # --------------------------------------------
    # No chair data
    # --------------------------------------------

    if not chair_row:
        return {
            "chair_score": None,
            "chair_stance": None,
            "communication_change": None,
            "statement_weight": 1.0,
            "chair_weight": 0.0,
            "chair_incremental_signal": False,
            "final_score": statement_score,
        }

    # --------------------------------------------
    # Chair analysis unavailable
    # --------------------------------------------

    if not chair_row.get(
        "analysis_available",
        False,
    ):
        return {
            "chair_score": None,
            "chair_stance": None,
            "communication_change": None,
            "statement_weight": 1.0,
            "chair_weight": 0.0,
            "chair_incremental_signal": False,
            "final_score": statement_score,
        }

    chair_score = to_float(
        chair_row.get(
            "press_conference_score"
        )
    )

    chair_stance = chair_row.get(
        "press_conference_signal"
    )

    communication_change = chair_row.get(
        "communication_change"
    )

    # --------------------------------------------
    # Score missing
    # --------------------------------------------

    if chair_score is None:
        return {
            "chair_score": None,
            "chair_stance": chair_stance,
            "communication_change":
                communication_change,
            "statement_weight": 1.0,
            "chair_weight": 0.0,
            "chair_incremental_signal": False,
            "final_score": statement_score,
        }

    # --------------------------------------------
    # Did Chair add incremental direction?
    # --------------------------------------------

    incremental = (
        communication_change
        in INCREMENTAL_CHANGES
    )

    if incremental:
        statement_weight = (
            INCREMENTAL_STATEMENT_WEIGHT
        )

        chair_weight = (
            INCREMENTAL_CHAIR_WEIGHT
        )

    else:
        statement_weight = (
            DEFAULT_STATEMENT_WEIGHT
        )

        chair_weight = (
            DEFAULT_CHAIR_WEIGHT
        )

    final_score = (
        statement_score
        * statement_weight
        +
        chair_score
        * chair_weight
    )

    return {
        "chair_score": chair_score,
        "chair_stance": chair_stance,
        "communication_change":
            communication_change,

        "statement_weight":
            statement_weight,

        "chair_weight":
            chair_weight,

        "chair_incremental_signal":
            incremental,

        "final_score": round(
            final_score,
            4,
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    policy_rows = load_json(
        POLICY_SIGNAL_PATH
    )

    chair_lookup = (
        load_chair_lookup()
    )

    # Only statements that actually exist
    policy_rows = [
        row
        for row in policy_rows
        if row.get("statement_found")
    ]

    policy_rows.sort(
        key=lambda x: x.get(
            "date",
            "",
        )
    )

    print(
        f"FOMC events: {len(policy_rows)}"
    )

    results = []

    previous_midpoint = None

    for row in policy_rows:
        date = row.get("date")

        print()
        print(f"[{date}]")

        # ====================================================
        # Actual policy decision
        # ====================================================

        target_range = (
            extract_target_range(row)
        )

        current_midpoint = (
            get_midpoint(target_range)
        )

        if (
            previous_midpoint is not None
            and current_midpoint is not None
        ):
            rate_change_bp = round(
                (
                    current_midpoint
                    - previous_midpoint
                )
                * 100
            )

        else:
            rate_change_bp = None

        # --------------------------------------------
        # Determine actual action
        # --------------------------------------------

        if rate_change_bp is not None:
            action = classify_action(
                rate_change_bp
            )

        else:
            raw_action = row.get(
                "policy_action"
            )

            action_map = {
                "TIGHTENING": "HIKE",
                "EASING": "CUT",
                "HOLD": "HOLD",
            }

            action = action_map.get(
                raw_action,
                "UNKNOWN",
            )

        # ====================================================
        # Statement signal
        # ====================================================

        statement_score = to_float(
            row.get("score")
        )

        if statement_score is None:
            statement_score = 0.0

        statement_stance = row.get(
            "stance",
            "UNKNOWN",
        )

        statement_change = (
            row.get("statement_change")
            or row.get("change")
        )

        statement_change_label_ko = (
            format_shift_label_ko(
                statement_change
            )
        )

        # ====================================================
        # Chair communication
        # ====================================================

        chair_row = (
            chair_lookup.get(date)
        )

        communication = (
            calculate_communication_signal(
                statement_score,
                chair_row,
            )
        )

        chair_change_label_ko = (
            format_shift_label_ko(
                communication[
                    "communication_change"
                ]
            )
        )

        # ====================================================
        # Final FOMC signal
        # ====================================================

        final_score = communication[
            "final_score"
        ]

        final_stance = classify_stance(
            final_score
        )

        event_label = (
            format_event_label(
                final_stance,
                action,
                rate_change_bp,
            )
        )

        event_label_ko = (
            format_event_label_ko(
                final_stance,
                action,
                rate_change_bp,
            )
        )

        # ====================================================
        # Target range
        # ====================================================

        if target_range:
            target_lower = target_range[0]
            target_upper = target_range[1]

        else:
            target_lower = None
            target_upper = None

        # ====================================================
        # Result
        # ====================================================

        result = {
            "date": date,
            "event_type": "FOMC",

            # ----------------------------------------
            # Actual decision
            # ----------------------------------------

            "previous_target_mid":
                previous_midpoint,

            "target_lower":
                target_lower,

            "target_upper":
                target_upper,

            "new_target_mid":
                current_midpoint,

            "rate_change_bp":
                rate_change_bp,

            "policy_action":
                action,

            # ----------------------------------------
            # Statement
            # ----------------------------------------

            "statement_stance":
                statement_stance,

            "statement_score":
                statement_score,

            "statement_change":
                statement_change,

            "statement_change_label_ko":
                statement_change_label_ko,

            "dissent":
                row.get("dissent"),

            # ----------------------------------------
            # Chair
            # ----------------------------------------

            "chair_stance":
                communication[
                    "chair_stance"
                ],

            "chair_score":
                communication[
                    "chair_score"
                ],

            "communication_change":
                communication[
                    "communication_change"
                ],

            "chair_change_label_ko":
                chair_change_label_ko,

            "chair_incremental_signal":
                communication[
                    "chair_incremental_signal"
                ],

            # ----------------------------------------
            # Weights
            # ----------------------------------------

            "statement_weight":
                communication[
                    "statement_weight"
                ],

            "chair_weight":
                communication[
                    "chair_weight"
                ],

            # ----------------------------------------
            # Final FOMC classification
            # ----------------------------------------

            "final_score":
                final_score,

            "final_stance":
                final_stance,

            "event_label":
                event_label,

            "event_label_ko":
                event_label_ko,

            # ----------------------------------------
            # Evidence
            # ----------------------------------------

            "statement_reason":
                row.get("reason"),

            "statement_evidence":
                row.get("key_evidence"),

            "chair_reason":
                (
                    chair_row.get("reason")
                    if chair_row
                    else None
                ),

            "chair_evidence":
                (
                    chair_row.get(
                        "key_evidence"
                    )
                    if chair_row
                    else None
                ),

            # ----------------------------------------
            # Sources
            # ----------------------------------------

            "statement_url":
                row.get(
                    "statement_url"
                ),

            "press_conference_url":
                (
                    chair_row.get(
                        "press_conference_url"
                    )
                    if chair_row
                    else None
                ),
        }

        results.append(result)

        # ====================================================
        # Console
        # ====================================================

        if target_range:
            print(
                "  target: "
                f"{target_lower:.3f}%"
                " - "
                f"{target_upper:.3f}%"
            )

        else:
            print(
                "  target: UNKNOWN"
            )

        print(
            f"  action: {action}"
            + (
                f" ({rate_change_bp:+}bp)"
                if rate_change_bp is not None
                else ""
            )
        )

        print(
            "  statement: "
            f"{statement_stance} "
            f"({statement_score:+.2f})"
        )

        print(
            "  statement change: "
            f"{statement_change}"
        )

        if communication[
            "chair_score"
        ] is not None:

            print(
                "  chair: "
                f"{communication['chair_stance']} "
                f"({communication['chair_score']:+.2f})"
            )

            print(
                "  chair change: "
                f"{communication['communication_change']}"
            )

        else:
            print(
                "  chair: NOT AVAILABLE"
            )

        print(
            "  weights: "
            f"statement "
            f"{communication['statement_weight']:.0%}"
            " / "
            f"chair "
            f"{communication['chair_weight']:.0%}"
        )

        print(
            "  FINAL: "
            f"{event_label_ko}"
            " / "
            f"{event_label} "
            f"({final_score:+.2f})"
        )

        # --------------------------------------------
        # Update rate baseline
        # --------------------------------------------

        if current_midpoint is not None:
            previous_midpoint = (
                current_midpoint
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