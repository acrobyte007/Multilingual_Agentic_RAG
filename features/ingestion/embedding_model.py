import os
import time
import re
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self):
        self.model = None
        self.BATCH_SIZE = 15
        self.REQUEST_DELAY = 1
        self.MAX_RETRIES = 3
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        self.model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self._initialized = True
        logger.info("EmbeddingService initialized")

    def tokenize_sentences(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return [s.strip().split() for s in sentences]

    def extract_retry_time(self, error_msg):
        match = re.search(r"retry in (\d+)", str(error_msg))
        return int(match.group(1)) if match else 60

    def embed_batch(self, batch_sentences, batch_id):
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"[BATCH {batch_id}] Attempt {attempt} | size={len(batch_sentences)}")
                vectors = self.model.embed_documents(batch_sentences)
                logger.info(f"[BATCH {batch_id}] SUCCESS")
                return vectors

            except Exception as e:
                logger.error(f"[BATCH {batch_id}] ERROR: {e}")

                if "429" in str(e):
                    wait_time = self.extract_retry_time(str(e))
                    logger.info(f"[BATCH {batch_id}] RATE LIMIT → sleeping {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise e

        raise Exception(f"[BATCH {batch_id}] FAILED after retries")

    def embed(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        total = len(sentences)
        logger.info(f"[INFO] Total sentences: {total}")

        tokens_list = self.tokenize_sentences(sentences)
        all_vectors = []

        batch_id = 0

        for i in range(0, total, self.BATCH_SIZE):
            batch_id += 1
            batch = sentences[i:i + self.BATCH_SIZE]

            logger.info(f"[INFO] Batch {batch_id} range {i}-{i+len(batch)-1}")

            vectors = self.embed_batch(batch, batch_id)
            all_vectors.extend(vectors)

            time.sleep(self.REQUEST_DELAY)

        vectors = np.array(all_vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms

        result = {}

        for i in range(total):
            result[i] = {
                "tokens": tokens_list[i],
                "embedding": normalized[i].tolist()
            }

        return result

embedding_service = EmbeddingService()