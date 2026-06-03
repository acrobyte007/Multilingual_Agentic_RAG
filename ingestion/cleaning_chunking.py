import re
import unicodedata
import asyncio
from typing import List, Dict
from langdetect import detect


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
    max_words: int = 150,
    overlap: int = 0
) -> List[Dict]:
    text = _clean_text(text)

    sentences = re.split(r'(?<=[.,?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_len = 0

    def finalize_chunk(chunk_sentences):
        chunk_text = " ".join(chunk_sentences).strip()
        words = chunk_text.split()
        lang_sample = " ".join(words[:50])
        lang = _detect_lang(lang_sample)
        return {"text": chunk_text, "lang": lang}

    for sentence in sentences:
        words = sentence.split()
        sentence_len = len(words)

        if sentence_len > max_words:
            for i in range(0, sentence_len, max_words):
                sub_words = words[i:i + max_words]
                chunk_text = " ".join(sub_words)
                lang_sample = " ".join(sub_words[:50])
                lang = _detect_lang(lang_sample)
                chunks.append({"text": chunk_text, "lang": lang})
            continue

        if current_len + sentence_len > max_words:
            chunks.append(finalize_chunk(current_chunk))

            if overlap > 0:
                current_chunk = current_chunk[-overlap:]
                current_len = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(finalize_chunk(current_chunk))

    return chunks


async def process_text(
    text: str,
    max_words: int = 150,
    overlap: int = 0
) -> List[Dict]:
    chunks = await asyncio.to_thread(
        _process_text_sync,
        text,
        max_words,
        overlap
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