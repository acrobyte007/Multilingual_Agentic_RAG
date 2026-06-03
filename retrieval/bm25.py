from logger.logger import get_logger
logger = get_logger(__name__)
import math
import re
import numpy as np
from typing import List, Dict
from collections import Counter
from ingestion.embedding_model import tokenize_sentences

def clean_token(token: str) -> bool:
    if not token or len(token) == 0:
        return False
    if not isinstance(token, str):
        return False
    if re.match(r'^[!@#$%^&*()_+\-=\[\]{};:\'",<>./?\\|`~]+$', token):
        return False
    if token == "#":
        return False
    if token == "_":
        return False
    return True

def filter_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    return [token for token in tokens if isinstance(token, str) and clean_token(token)]

def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if np.max(scores_array) == np.min(scores_array):
        return [0.5] * len(scores) if scores_array[0] != 0 else [0.0] * len(scores)
    
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    if min_score < 0:
        shifted_scores = scores_array - min_score
        max_shifted = np.max(shifted_scores)
        if max_shifted > 0:
            normalized = shifted_scores / max_shifted
        else:
            normalized = np.zeros_like(scores_array)
    else:
        normalized = (scores_array - min_score) / (max_score - min_score)
    
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    
    return normalized.tolist()

async def bm25_rerank(
    chunks: List[str],
    chunks_tokens: List[List[str]],
    chunk_ids: List[str],
    chunk_languages: List[str],
    translated_queries: Dict[str, str]
) -> List[str]:
    logger.info(f"Reranking {len(chunks_tokens)} chunks with BM25")
    
    query_tokens_by_lang = {}
    
    for lang, translated_query in translated_queries.items():
        if translated_query:
            translated_tokens = tokenize_sentences([translated_query])
            query_tokens_by_lang[lang] = filter_tokens(translated_tokens[0])
            logger.info(f"Loaded {len(query_tokens_by_lang[lang])} tokens for language: {lang}")
    
    filtered_chunks_tokens = [filter_tokens(doc) for doc in chunks_tokens]
    
    doc_lengths = [len(doc) for doc in filtered_chunks_tokens]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    
    idf = {}
    
    for doc_tokens in filtered_chunks_tokens:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            idf[term] = idf.get(term, 0) + 1
    
    num_docs = len(filtered_chunks_tokens)
    for term in idf:
        idf[term] = math.log((num_docs - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
    
    bm25_scores = []
    k1, b = 1.5, 0.75
    
    for idx, doc_tokens in enumerate(filtered_chunks_tokens):
        score = 0
        doc_length = doc_lengths[idx]
        term_freqs = Counter(doc_tokens)
        chunk_lang = chunk_languages[idx] if idx < len(chunk_languages) else "en"
        
        if chunk_lang in query_tokens_by_lang:
            active_query_tokens = query_tokens_by_lang[chunk_lang]
        else:
            active_query_tokens = query_tokens_by_lang.get("en", [])
        
        for term in active_query_tokens:
            if term not in idf:
                continue
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            term_score = idf[term] * (numerator / denominator)
            score += term_score
        bm25_scores.append(score)
    
    if bm25_scores:
        logger.info(f"BM25 scores range: [{min(bm25_scores):.4f}, {max(bm25_scores):.4f}]")
    
    normalized_scores = normalize_scores(bm25_scores)
    
    results = []
    for i in range(len(chunks)):
        results.append({
            "chunk_id": chunk_ids[i],
            "chunk_text": chunks[i],
            "bm25_score": normalized_scores[i] if normalized_scores else 0.0,
            "language": chunk_languages[i] if i < len(chunk_languages) else "en"
        })
    
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    results = results[:5]
    
    reranked_chunks = [r["chunk_text"] for r in results]
    
    if results:
        logger.info(f"Reranking complete. Top score: {results[0]['bm25_score']:.4f}, Language: {results[0]['language']}")
    
    return reranked_chunks