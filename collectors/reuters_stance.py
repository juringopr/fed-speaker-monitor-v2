from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


REUTERS_STANCE_URL = (
    "https://graphics.thomsonreuters.com/testfiles/2025/1wtMSweut5YR/"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "results"
    / "reuters_stance.json"
)

VALID_STANCES = {
    "dove",
    "dovish",
    "centrist",
    "hawkish",
    "hawk",
}


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _load_existing() -> dict:
    if not OUTPUT_PATH.exists():
        return {}

    try:
        return json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def collect_reuters_stance() -> dict:
    """
    Reuters 'Doves and Hawks' 페이지에서
    현재 Fed 인사별 stance를 가져온다.

    stance:
        DOVE
        DOVISH
        CENTRIST
        HAWKISH
        HAWK
    """

    response = requests.get(
        REUTERS_STANCE_URL,
        timeout=20,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    members = {}

    for block in soup.select(".board-member"):

        name_el = block.select_one(".name")
        head_el = block.select_one("img.hedshot")

        if not name_el or not head_el:
            continue

        name = _clean_text(name_el.get_text())

        classes = head_el.get("class", [])

        stance = next(
            (
                c
                for c in classes
                if c.lower() in VALID_STANCES
            ),
            None,
        )

        if not name or not stance:
            continue

        # -----------------------------
        # title
        # -----------------------------

        title_el = block.select_one(".title")

        title = (
            _clean_text(title_el.get_text())
            if title_el
            else None
        )

        # -----------------------------
        # voting status
        # -----------------------------

        status_el = block.select_one(".status")

        voting_status = (
            _clean_text(status_el.get_text())
            if status_el
            else None
        )

        # -----------------------------
        # Reuters quote
        # -----------------------------

        quote_el = block.select_one(".quote-text")

        quote = (
            _clean_text(quote_el.get_text())
            if quote_el
            else None
        )

        # -----------------------------
        # quote date + Reuters article
        # -----------------------------

        quote_date = None
        article_url = None

        quote_date_el = block.select_one(".quote-date")

        if quote_date_el:

            quote_date = _clean_text(
                quote_date_el.get_text()
            )

            link = quote_date_el.select_one("a")

            if link:
                article_url = link.get("href")

        members[name] = {
            "stance": stance.upper(),
            "title": title,
            "voting_status": voting_status,
            "quote": quote,
            "quote_date": quote_date,
            "article_url": article_url,
        }

    if not members:
        raise RuntimeError(
            "Reuters stance parsing returned 0 members."
        )

    # 페이지 자체 업데이트 시각
    updated_at = None

    for time_el in soup.select("time[datetime]"):
        parent_text = _clean_text(
            time_el.parent.get_text()
            if time_el.parent
            else ""
        )

        if "Last updated" in parent_text:
            updated_at = time_el.get("datetime")
            break

    return {
        "source": "Reuters",
        "source_url": REUTERS_STANCE_URL,
        "reuters_updated_at": updated_at,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "member_count": len(members),
        "members": members,
    }


def update_reuters_stance() -> dict:
    """
    정상적으로 파싱됐을 때만 JSON을 교체한다.

    Reuters 접속/파싱 실패 시
    기존 JSON을 그대로 유지한다.
    """

    existing = _load_existing()

    try:
        data = collect_reuters_stance()

        member_count = data.get(
            "member_count",
            0,
        )

        # Reuters 페이지 구조 변경 등으로
        # 비정상적으로 적게 잡히면 기존 파일 보호
        if member_count < 10:
            raise RuntimeError(
                f"Only {member_count} Reuters members parsed."
            )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[REUTERS] updated: "
            f"{member_count} members"
        )

        if data.get("reuters_updated_at"):
            print(
                "[REUTERS] page updated:",
                data["reuters_updated_at"],
            )

        return data

    except Exception as exc:

        print(
            f"[REUTERS] update failed: {exc}"
        )

        if existing:
            print(
                "[REUTERS] keeping existing JSON"
            )
            return existing

        return {}


def load_reuters_stance() -> dict:
    """
    app.py 등에서 저장된 Reuters stance를 읽을 때 사용.
    """

    return _load_existing()


if __name__ == "__main__":

    data = update_reuters_stance()

    members = data.get(
        "members",
        {},
    )

    print()

    for name, info in members.items():

        print(
            f"{name:25} "
            f"{info.get('stance', ''):10} "
            f"{info.get('voting_status', '')}"
        )