import re
import unicodedata
import asyncio
from typing import List, Dict

from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def _clean_text(text: str) -> str:
    if not text:
        return text

    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"(\.\s*){3,}", " ", text)
    text = re.sub(r"https:\s+", "https://", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _process_text_sync(
    text: str,
    max_words: int = 250,
    overlap: int = 50
) -> List[Dict]:
    text = _clean_text(text)

    chunk_size = max_words * 6
    chunk_overlap = overlap * 6

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            " ",
            ""
        ],
        is_separator_regex=False,
    )

    split_texts = splitter.split_text(text)

    chunks = []

    for chunk in split_texts:
        chunk = chunk.strip()
        if not chunk:
            continue

        lang_sample = " ".join(chunk.split()[:50])

        chunks.append({
            "text": chunk,
            "lang": _detect_lang(lang_sample)
        })

    return chunks


async def process_text(
    text: str,
    max_words: int = 300,
    overlap: int = 50
) -> List[Dict]:
    chunks = await asyncio.to_thread(
        _process_text_sync,
        text,
        max_words,
        overlap,
    )

    result = []

    for i, chunk in enumerate(chunks):
        result.append({
            "chunk_id": i + 1,
            "text": chunk["text"],
            "language": chunk["lang"],
            "word_count": len(chunk["text"].split())
        })

    return result