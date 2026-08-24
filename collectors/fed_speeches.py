import requests
from bs4 import BeautifulSoup

from fed_speaker_monitor_v2.config import (
    REQUEST_TIMEOUT,
    USER_AGENT,
)
from fed_speaker_monitor_v2.models import Document


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": USER_AGENT,
}


def _fetch_page(url: str) -> BeautifulSoup | None:
    """
    Fed speech/testimony 페이지를 가져온다.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return BeautifulSoup(
            response.text,
            "html.parser",
        )

    except requests.RequestException:
        return None


# ============================================================
# Article
# ============================================================

def _find_article(soup: BeautifulSoup):
    """
    Fed speech/testimony의 실제 article 영역을 찾는다.

    현재 Fed 페이지 구조:
        div#content
            -> div#article
                -> speech content
    """

    article = soup.find(
        "div",
        id="article",
    )

    if article is not None:
        return article

    return soup.select_one(
        "div.col-xs-12.col-sm-8.col-md-8"
    )


# ============================================================
# Speaker
# ============================================================

def _extract_speaker(soup: BeautifulSoup) -> str | None:
    """
    Fed speech/testimony 페이지에서 speaker 이름을 추출한다.

    예:
        Governor Lisa D. Cook
        Chair Jerome H. Powell
        Vice Chair Philip N. Jefferson

    반환:
        Lisa D. Cook
        Jerome H. Powell
        Philip N. Jefferson
    """

    article = _find_article(soup)

    if article is None:
        return None

    titles = (
        "Vice Chair for Supervision ",
        "Vice Chair ",
        "Chairman ",
        "Chair ",
        "Governor ",
    )

    text_elements = article.find_all(
        ["p", "div", "h3", "h4"]
    )

    for element in text_elements:

        text = element.get_text(
            " ",
            strip=True,
        )

        for title in titles:

            if text.startswith(title):
                name = text[len(title):].strip()

                # 부모 div 전체 텍스트가 잡히는 경우 방지
                if len(name) < 100:
                    return name

    return None


# ============================================================
# Speech Text
# ============================================================

def _extract_text(soup: BeautifulSoup) -> str:
    """
    Fed speech/testimony의 article 전체 텍스트를 추출한다.

    Fed 페이지는 본문이 반드시 <p>로만 구성되어 있지 않기 때문에
    #article 전체 텍스트를 사용한다.
    """

    article = _find_article(soup)

    if article is None:
        return ""

    text = article.get_text(
        "\n",
        strip=True,
    )

    return text


# ============================================================
# Single Document
# ============================================================

def enrich_fed_document(
    document: Document,
) -> Document:
    """
    fed_official.py에서 생성된 Document에
    speaker와 text를 추가한다.
    """

    soup = _fetch_page(document.url)

    if soup is None:
        document.fetch_ok = False
        return document

    speaker = _extract_speaker(soup)
    text = _extract_text(soup)

    document.speaker = speaker
    document.text = text

    if not text:
        document.fetch_ok = False

    return document


# ============================================================
# Multiple Documents
# ============================================================

def enrich_fed_documents(
    documents: list[Document],
) -> list[Document]:
    """
    여러 Fed Document의 본문과 speaker를 수집한다.
    """

    results = []

    for document in documents:

        enriched = enrich_fed_document(
            document
        )

        results.append(enriched)

    return results