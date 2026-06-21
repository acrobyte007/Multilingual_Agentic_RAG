import os
import json
import logging
from typing import List, Dict, Any
from pinecone import Pinecone, PineconeException
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeService:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self._pc = None
        self._index = None
        self._initialized = False

    def initialize(self):
        """Lazy init (prevents double import issues)"""
        if self._initialized:
            return

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = "index"

        if not api_key:
            raise Exception("Missing PINECONE_API_KEY")

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

        self._initialized = True
        logger.info("Pinecone initialized once")

    def upsert(
        self,
        namespace: str,
        document_id: str,
        vectors: List[List[float]],
        chunks: List[str],
        lang_list: List[str],
        tokens_list: List[List[str]]
    ) -> Dict[str, Any]:

        self.initialize()

        if not (len(vectors) == len(chunks) == len(lang_list) == len(tokens_list)):
            raise ValueError("Input lists must have same length")

        vectors_to_upsert = []

        for i, (vector, chunk, lang, tokens) in enumerate(
            zip(vectors, chunks, lang_list, tokens_list), 1
        ):
            vectors_to_upsert.append({
                "id": f"{document_id}#chunk{i}",
                "values": vector,
                "metadata": {
                    "document_id": document_id,
                    "chunk_number": i,
                    "chunk_text": chunk,
                    "language": lang,
                    "tokens": json.dumps(tokens)
                }
            })

        responses = []

        for i in range(0, len(vectors_to_upsert), self.batch_size):
            batch = vectors_to_upsert[i:i + self.batch_size]
            try:
                res = self._index.upsert(namespace=namespace, vectors=batch)
                responses.append(res)
            except Exception as e:
                logger.error(f"Upsert failed: {e}")
                raise

        return {"batches": len(responses)}

    def search(self, namespace: str, vector: List[float], doc_ids: List[str], top_k: int):

        self.initialize()

        all_matches = []

        for doc_id in doc_ids:
            try:
                result = self._index.query(
                    namespace=namespace,
                    vector=vector,
                    top_k=top_k,
                    filter={"document_id": {"$eq": doc_id}},
                    include_metadata=True
                )

                matches = result.get("matches", [])
                all_matches.extend(matches)

            except PineconeException as e:
                logger.error(f"Pinecone error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

        chunk_texts, chunk_ids, tokens_list, lang_list = [], [], [], []

        for match in all_matches:
            metadata = match.get("metadata", {})

            chunk_texts.append(metadata.get("chunk_text", ""))
            chunk_ids.append(match.get("id", ""))
            lang_list.append(metadata.get("language", "unknown"))

            tokens_str = metadata.get("tokens", "")
            try:
                tokens = json.loads(tokens_str) if tokens_str else []
            except:
                tokens = []

            tokens_list.append(tokens)

        return {
            "chunk_texts": chunk_texts,
            "chunk_ids": chunk_ids,
            "tokens_list": tokens_list,
            "lang_list": lang_list
        }

    def delete(self, namespace: str, document_id: str):

        self.initialize()

        try:
            self._index.delete(
                namespace=namespace,
                filter={"document_id": {"$eq": document_id}}
            )
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise


pinecone_service = PineconeService()