from logger.logger import get_logger
import dotenv
import numpy as np
import time
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
logger = get_logger(__name__)
dotenv.load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

BATCH_SIZE = 15
REQUEST_DELAY =1
MAX_RETRIES = 3


def tokenize_sentences(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    return [s.strip().split() for s in sentences]


def extract_retry_time(error_msg):
    match = re.search(r"retry in (\d+)", str(error_msg))
    if match:
        return int(match.group(1))
    return 60


def embed_batch(batch_sentences, batch_id):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[BATCH {batch_id}] Attempt {attempt} | size={len(batch_sentences)}")
            vectors = embedding_model.embed_documents(batch_sentences)
            logger.info(f"[BATCH {batch_id}] SUCCESS")
            return vectors

        except Exception as e:
            logger.info(f"[BATCH {batch_id}] ERROR: {e}")
            if "429" in str(e):
                wait_time = extract_retry_time(str(e))
                logger.info(f"[BATCH {batch_id}] RATE LIMIT → sleeping {wait_time}s")
                time.sleep(wait_time)
            else:
                raise e

    raise Exception(f"[BATCH {batch_id}] FAILED after retries")


def embedding_process(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]

    total = len(sentences)
    logger.info(f"[INFO] Total sentences: {total}")

    tokens_list = tokenize_sentences(sentences)
    all_vectors = []

    batch_id = 0

    for i in range(0, total, BATCH_SIZE):
        batch_id += 1
        batch = sentences[i:i + BATCH_SIZE]

        logger.info(f"[INFO] Batch {batch_id} range {i}-{i+len(batch)-1}")

        vectors = embed_batch(batch, batch_id)
        all_vectors.extend(vectors)

        time.sleep(REQUEST_DELAY)

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

