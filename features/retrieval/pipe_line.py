import logging
from typing import List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from features.retrieval.semantic_search import retrieve
from features.retrieval.bm25 import bm25_rerank

async def top_k_retrieval(name_space: str, user_query: str, doc_ids: List[str], translated_queries: dict = None):
    search_result = await retrieve(name_space, user_query, doc_ids, top_k=30)
    chunk_texts = search_result["chunk_texts"]
    chunk_ids = search_result["chunk_ids"]
    tokens_list = search_result["tokens_list"]
    language_list = search_result["lang_list"]
    logger.info(f"language_list: {language_list}")
    
    reranked_chunks = await bm25_rerank(
        chunks=chunk_texts,
        chunks_tokens=tokens_list,
        chunk_ids=chunk_ids,
        chunk_languages=language_list,
        translated_queries=translated_queries if translated_queries else {}
    )
    
    return reranked_chunks

