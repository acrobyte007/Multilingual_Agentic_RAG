import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import List
from database.vector_database import pinecone_service
from services.embedding_model import embedding_service
async def retrieve(name_space: str, user_query: str, doc_ids: List[str], top_k: int = 10):
        logger.info(f"Retrieving for namespace: {name_space}, query: {user_query}, doc_ids: {doc_ids}")
        vector = embedding_service.embed(user_query)
        logger.info(f"Generated vector of length {len(vector)}")
        search_result=pinecone_service.search(
            namespace=name_space,
            vector=vector[0]["embedding"],
            doc_ids=doc_ids,
            top_k=top_k
        )
        logger.info(f"Retrieved {len(search_result['chunk_texts'])} chunks")
        return search_result

