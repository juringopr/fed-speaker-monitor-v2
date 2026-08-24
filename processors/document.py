import re

from fed_speaker_monitor_v2.models import Document


# ============================================================
# Speaker Aliases
# ============================================================

SPEAKER_ALIASES = {
    "Jerome H. Powell": "Jerome Powell",
    "Philip N. Jefferson": "Philip Jefferson",
    "Michelle W. Bowman": "Michelle Bowman",
    "Michael S. Barr": "Michael Barr",
    "Lisa D. Cook": "Lisa Cook",
    "Christopher J. Waller": "Christopher Waller",
    "Adriana D. Kugler": "Adriana Kugler",
    "John C. Williams": "John Williams",
    "Patrick T. Harker": "Patrick Harker",
    "Thomas I. Barkin": "Thomas Barkin",
    "Raphael W. Bostic": "Raphael Bostic",
    "Austan D. Goolsbee": "Austan Goolsbee",
    "Alberto G. Musalem": "Alberto Musalem",
    "Neel Kashkari": "Neel Kashkari",
    "Jeffrey R. Schmid": "Jeffrey Schmid",
    "Lorie K. Logan": "Lorie Logan",
    "Mary C. Daly": "Mary Daly",
}


# ============================================================
# Speaker
# ============================================================

def normalize_speaker(
    speaker: str | None,
) -> str | None:
    """
    Fed 페이지의 정식 이름을 프로젝트 내부 canonical name으로 변환.

    예:
        Lisa D. Cook
        -> Lisa Cook
    """

    if not speaker:
        return None

    speaker = speaker.strip()

    return SPEAKER_ALIASES.get(
        speaker,
        speaker,
    )


# ============================================================
# Text
# ============================================================

def _normalize_whitespace(text: str) -> str:
    """
    과도한 공백과 줄바꿈을 정리한다.
    """

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _remove_fed_header(
    text: str,
    title: str,
    speaker: str | None,
) -> str:
    """
    Fed speech 앞부분의 metadata를 제거한다.

    예:
        August 05, 2026
        Outlook for the U.S. and Alaskan Economies
        Governor Lisa D. Cook
        At the ...
        Share
        Thank you...

    위 metadata를 제거하고 실제 speech 시작 부분부터 반환한다.
    """

    if not text:
        return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # "Share"가 있으면 그 다음부터 실제 speech로 간주
    for index, line in enumerate(lines):

        if line.lower() == "share":

            return "\n\n".join(
                lines[index + 1:]
            ).strip()

    # Share가 없는 페이지는 원문 보존
    return text


def clean_document_text(
    document: Document,
) -> str:
    """
    source에 따라 Document text를 정제한다.
    """

    text = document.text or ""

    text = _normalize_whitespace(text)

    if document.source in {
        "fed_speech",
        "fed_testimony",
    }:

        text = _remove_fed_header(
            text=text,
            title=document.title,
            speaker=document.speaker,
        )

    return text


# ============================================================
# Document Processor
# ============================================================

def process_document(
    document: Document,
) -> Document:
    """
    하나의 Document를 정규화한다.

    처리:
        1. speaker canonicalization
        2. text cleanup
    """

    document.text = clean_document_text(
        document
    )

    document.speaker = normalize_speaker(
        document.speaker
    )

    return document


# ============================================================
# Multiple Documents
# ============================================================

def process_documents(
    documents: list[Document],
) -> list[Document]:
    """
    여러 Document를 정규화한다.
    """

    return [
        process_document(document)
        for document in documents
    ]