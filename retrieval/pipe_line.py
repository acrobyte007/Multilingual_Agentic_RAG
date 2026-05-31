import logging
from typing import List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from retrieval.semantic_search import retrieve
from retrieval.bm25 import bm25_rerank
from ingestion.embedding_model import tokenize_sentences
async def top_k_retrieval(name_space: str, user_query: str, doc_ids: List[str], top_k: int = 20):
    search_result = await retrieve(name_space, user_query, doc_ids, top_k=top_k)
    chunk_texts = search_result["chunk_texts"]
    chunk_ids = search_result["chunk_ids"]
    tokens_list = search_result["tokens_list"]
    query_tokens = tokenize_sentences([user_query])
    reranked_chunks = await bm25_rerank(query_tokens[0], chunk_texts, tokens_list, chunk_ids)
    return reranked_chunks

