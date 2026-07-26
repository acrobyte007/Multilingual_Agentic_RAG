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
    overlap: int = 25
) -> List[Dict]:
    text = _clean_text(text)

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        if sentence_word_count > max_words:
            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                lang_sample = " ".join(chunk_text.split()[:50])
                chunks.append({
                    "text": chunk_text,
                    "lang": _detect_lang(lang_sample)
                })
                current_sentences = []
                current_word_count = 0

            words = sentence_words
            i = 0
            step = max_words - overlap if max_words > overlap else max_words

            while i < len(words):
                chunk_words = words[i:i + max_words]
                chunk_text = " ".join(chunk_words)

                lang_sample = " ".join(chunk_words[:50])
                chunks.append({
                    "text": chunk_text,
                    "lang": _detect_lang(lang_sample)
                })

                i += step

            continue

        if current_word_count + sentence_word_count <= max_words:
            current_sentences.append(sentence)
            current_word_count += sentence_word_count
        else:
            # Save current chunk
            chunk_text = " ".join(current_sentences).strip()
            lang_sample = " ".join(chunk_text.split()[:50])

            chunks.append({
                "text": chunk_text,
                "lang": _detect_lang(lang_sample)
            })
            if overlap > 0:
                overlap_words = chunk_text.split()[-overlap:]
                overlap_text = " ".join(overlap_words)
                current_sentences = [overlap_text, sentence]
                current_word_count = len(overlap_words) + sentence_word_count
            else:
                current_sentences = [sentence]
                current_word_count = sentence_word_count

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        lang_sample = " ".join(chunk_text.split()[:50])

        chunks.append({
            "text": chunk_text,
            "lang": _detect_lang(lang_sample)
        })

    return chunks


async def process_text(
    text: str,
    max_words: int = 150,
    overlap: int = 25
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