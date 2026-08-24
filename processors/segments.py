import hashlib
import re

from fed_speaker_monitor_v2.config import (
    MIN_SEGMENT_LENGTH,
    MAX_SEGMENT_LENGTH,
)
from fed_speaker_monitor_v2.models import (
    Document,
    Segment,
)


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(paragraph: str, max_length: int) -> list[str]:
    if len(paragraph) <= max_length:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"
        if len(candidate) <= max_length:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def _build_segment_texts(text: str, max_length: int = MAX_SEGMENT_LENGTH) -> list[str]:
    paragraphs = _split_paragraphs(text)
    expanded_paragraphs = []

    for paragraph in paragraphs:
        expanded_paragraphs.extend(
            _split_long_paragraph(paragraph, max_length)
        )

    segments = []
    current = []
    current_length = 0

    for paragraph in expanded_paragraphs:
        separator_length = 2 if current else 0
        candidate_length = current_length + separator_length + len(paragraph)

        if current and candidate_length > max_length:
            segments.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length = candidate_length

    if current:
        segments.append("\n\n".join(current))

    return segments


def _document_key(document: Document) -> str:
    """URL 기반 짧은 문서 식별자. 같은 speaker의 다른 문서 segment 충돌 방지."""
    raw = document.url or f"{document.source}|{document.title}|{document.published_at or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _speaker_key(speaker: str | None) -> str:
    value = (speaker or "unknown").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or "unknown"


def segment_document(document: Document) -> list[Segment]:
    if not document.text:
        return []

    segment_texts = _build_segment_texts(document.text)
    segments = []
    segment_number = 1
    document_key = _document_key(document)
    speaker_key = _speaker_key(document.speaker)

    for text in segment_texts:
        if len(text) < MIN_SEGMENT_LENGTH:
            continue

        segment_id = f"{speaker_key}_{document_key}_{segment_number:03d}"

        segments.append(
            Segment(
                segment_id=segment_id,
                document_url=document.url,
                text=text,
                speaker=document.speaker,
            )
        )
        segment_number += 1

    return segments


def segment_documents(documents: list[Document]) -> list[Segment]:
    segments = []
    for document in documents:
        segments.extend(segment_document(document))
    return segments
