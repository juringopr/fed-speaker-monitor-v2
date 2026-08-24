import json
from pathlib import Path
from collections import Counter


# ======================================================
# 경로
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

PATH = (
    BASE_DIR
    / "data"
    / "results"
    / "news_history.json"
)


# ======================================================
# Evidence Quality Weight
# ======================================================

def get_quality_weight(row):
    """
    Speaker evidence 품질에 따라 가중치를 부여한다.

    HIGH   = 1.00
    MEDIUM = 0.80
    LOW    = 0.50
    REJECT = 0.00
    """

    evidence_type = row.get("speaker_evidence_type")
    directness = row.get("directness")
    confidence = row.get("evidence_confidence")
    policy_phrase = row.get("policy_bearing_phrase")

    # 실제 speaker evidence가 아님
    if evidence_type in {
        "NONE",
        "CONTEXT_ONLY",
    }:
        return 0.0

    # 직접 인용
    if evidence_type == "DIRECT_QUOTE":
        return 1.0

    # Speaker에게 직접 귀속된 paraphrase
    if (
        evidence_type == "ATTRIBUTED_PARAPHRASE"
        and directness == "DIRECT"
    ):
        if confidence == "HIGH":
            return 1.0

        return 0.80

    # 간접 attributed paraphrase
    if evidence_type == "ATTRIBUTED_PARAPHRASE":
        return 0.80

    # 언론이 전달한 speaker position
    # 완전히 제거하지 않고 낮은 가중치 부여
    if evidence_type == "REPORTED_POSITION":
        if policy_phrase:
            return 0.50

    return 0.0


# ======================================================
# 데이터 로드
# ======================================================

print(f"Loading: {PATH}")

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# 새 detailed schema가 존재하는 행만 사용
detailed = [
    row
    for row in data
    if "speaker_evidence_type" in row
]


# ======================================================
# 가중치 계산
# ======================================================

results = []

for row in detailed:

    try:
        raw_score = float(
            row.get("score") or 0
        )

    except (TypeError, ValueError):
        raw_score = 0.0

    weight = get_quality_weight(row)

    weighted_score = (
        raw_score * weight
    )

    results.append(
        {
            "speaker": row.get("speaker"),
            "stance": row.get("stance"),

            "raw_score": raw_score,
            "weight": weight,
            "weighted_score": weighted_score,

            "evidence_type": row.get(
                "speaker_evidence_type"
            ),

            "directness": row.get(
                "directness"
            ),

            "confidence": row.get(
                "evidence_confidence"
            ),

            "signal_strength": row.get(
                "signal_strength"
            ),

            "policy_phrase": row.get(
                "policy_bearing_phrase"
            ),

            "title": row.get(
                "title",
                ""
            ),
        }
    )


# ======================================================
# 전체 분포
# ======================================================

print()
print("=" * 100)
print("QUALITY WEIGHT SIMULATION")
print("=" * 100)

print(
    f"Total history  : "
    f"{len(data)}"
)

print(
    f"Detailed rows  : "
    f"{len(results)}"
)

print(
    "Weights        :",
    dict(
        sorted(
            Counter(
                row["weight"]
                for row in results
            ).items()
        )
    ),
)

print()


# ======================================================
# Hawk / Dove Signal
# ======================================================

usable = [
    row
    for row in results
    if row["weight"] > 0
    and row["stance"] in {
        "HAWKISH",
        "DOVISH",
    }
]


print(
    f"Usable signals : "
    f"{len(usable)}"
)

print()


# ======================================================
# Weight별 usable 개수
# ======================================================

usable_weights = Counter(
    row["weight"]
    for row in usable
)

print(
    "Usable weights :",
    dict(
        sorted(
            usable_weights.items()
        )
    ),
)

print()


# ======================================================
# 개별 결과
# ======================================================

for i, row in enumerate(
    usable,
    start=1,
):

    print(
        f"[{i}/{len(usable)}] "
        f"{row['speaker']}"
    )

    print(
        f"    {row['stance']}"
        f" | raw={row['raw_score']:+.2f}"
        f" | weight={row['weight']:.2f}"
        f" | final={row['weighted_score']:+.3f}"
    )

    print(
        f"    {row['evidence_type']}"
        f" | {row['directness']}"
        f" | {row['confidence']}"
        f" | {row['signal_strength']}"
    )

    print(
        f"    PHRASE: "
        f"{row['policy_phrase']}"
    )

    print(
        f"    TITLE: "
        f"{row['title']}"
    )

    print()


# ======================================================
# 평균 비교
# ======================================================

print("=" * 100)

if usable:

    raw_avg = (
        sum(
            row["raw_score"]
            for row in usable
        )
        / len(usable)
    )

    weighted_avg = (
        sum(
            row["weighted_score"]
            for row in usable
        )
        / len(usable)
    )

    print(
        f"Raw average      : "
        f"{raw_avg:+.3f}"
    )

    print(
        f"Weighted average : "
        f"{weighted_avg:+.3f}"
    )

print("=" * 100)