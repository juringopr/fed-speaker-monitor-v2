import json
from pathlib import Path

from fed_speaker_monitor_v2.llm.stance import analyze_segment
from fed_speaker_monitor_v2.models import Segment


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "data" / "results"

HISTORY_PATH = RESULTS_DIR / "news_history.json"
BACKUP_PATH = RESULTS_DIR / "news_history_before_strong_backfill.json"

STRONG_THRESHOLD = 0.70


DETAIL_FIELDS = {
    "content_type",
    "directness",
    "temporal",
    "uncertainty",
    "policy_action",
    "signal_strength",
    "policy_relevance_score",
    "speaker_evidence_type",
    "policy_bearing_phrase",
    "stance_driver",
    "bis_stance",
    "evidence_confidence",
    "text_sufficiency",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def needs_backfill(row):
    """
    기존 Strong Signal 후보 중
    아직 새 detailed schema가 없는 행만 선택.
    """

    # 이미 상세 분석된 행은 재분석하지 않음
    if DETAIL_FIELDS.issubset(row.keys()):
        return False

    score = row.get("score")

    if score is None:
        return False

    try:
        score = float(score)
    except (TypeError, ValueError):
        return False

    # 기존 strong signal만 대상
    if abs(score) < STRONG_THRESHOLD:
        return False

    if not row.get("policy_relevant"):
        return False

    if row.get("stance") == "IRRELEVANT":
        return False

    if not row.get("evidence"):
        return False

    return True


def make_segment(row):
    """
    news_history 행을 analyze_segment()가
    받을 수 있는 Segment 객체로 변환.
    """

    return Segment(
        segment_id=row["segment_id"],
        document_url=row.get("url", ""),
        text=row.get("summary", ""),
        speaker=row.get("speaker", ""),
    )


def update_row(row, result):
    """
    새 stance 분석 결과를 기존 history 행에 반영.
    """

    row["policy_relevant"] = result.policy_relevant
    row["stance"] = result.stance
    row["score"] = result.score

    row["content_type"] = result.content_type
    row["directness"] = result.directness
    row["temporal"] = result.temporal
    row["uncertainty"] = result.uncertainty

    row["evidence"] = result.evidence
    row["reasoning"] = result.reasoning

    row["policy_action"] = result.policy_action
    row["signal_strength"] = result.signal_strength
    row["policy_relevance_score"] = result.policy_relevance_score

    row["speaker_evidence_type"] = result.speaker_evidence_type
    row["policy_bearing_phrase"] = result.policy_bearing_phrase
    row["stance_driver"] = result.stance_driver

    row["bis_stance"] = result.bis_stance
    row["evidence_confidence"] = result.evidence_confidence
    row["text_sufficiency"] = result.text_sufficiency


def run_backfill():
    history = load_json(HISTORY_PATH)

    candidates = [
        row
        for row in history
        if needs_backfill(row)
    ]

    print(f"Total history: {len(history)}")
    print(
        f"Strong candidates requiring backfill: "
        f"{len(candidates)}"
    )

    if not candidates:
        print("Nothing to backfill.")
        return

    # 최초 실행 시 원본 백업 1회 생성
    if not BACKUP_PATH.exists():
        write_json(
            BACKUP_PATH,
            history,
        )

        print(
            f"Backup saved: {BACKUP_PATH}"
        )
    else:
        print(
            f"Backup already exists: {BACKUP_PATH}"
        )

    total = len(candidates)
    success = 0
    failed = 0

    for i, row in enumerate(
        candidates,
        start=1,
    ):
        print()
        print(
            f"[{i}/{total}] "
            f"{row.get('speaker')} | "
            f"{row.get('segment_id')}"
        )

        try:
            segment = make_segment(row)

            result = analyze_segment(
                segment
            )

            update_row(
                row,
                result,
            )

            # 1건마다 저장.
            # 중간 종료돼도 완료된 결과 보존.
            write_json(
                HISTORY_PATH,
                history,
            )

            success += 1

            print(
                f"    -> "
                f"{result.stance} "
                f"{result.score:+.2f}"
            )

            print(
                f"       evidence_type="
                f"{result.speaker_evidence_type}"
            )

            print(
                f"       directness="
                f"{result.directness}"
            )

            print(
                f"       signal_strength="
                f"{result.signal_strength}"
            )

            print(
                f"       confidence="
                f"{result.evidence_confidence}"
            )

        except Exception as e:
            failed += 1

            print(
                f"    ERROR: "
                f"{type(e).__name__}: {e}"
            )

    print()
    print("=" * 80)
    print("BACKFILL DONE")
    print(f"Candidates : {total}")
    print(f"Success    : {success}")
    print(f"Failed     : {failed}")
    print(f"Updated    : {HISTORY_PATH}")
    print(f"Backup     : {BACKUP_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_backfill()