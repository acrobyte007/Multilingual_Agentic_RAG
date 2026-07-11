import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from features.ingestion.extraction import extract_text
from features.ingestion.cleaning_chunking import process_text
from services.embedding_model import embedding_service
from database.vector_database import pinecone_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    
    def __init__(self):
        self.max_words_per_chunk = 250
        self.chunk_overlap = 50
        self.batch_size = 100
        pinecone_service.batch_size = 100
    
    async def ingest_document(
        self,
        namespace: str,
        file_path: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        
        if document_id is None:
            document_id = Path(file_path).stem
        
        logger.info(f"Starting ingestion for namespace {namespace}, document: {document_id}")
        
        try:
            raw_text = await extract_text(file_path)
            
            if not raw_text or len(raw_text.strip()) == 0:
                raise ValueError(f"No text content extracted from {file_path}")
            
            logger.info(f"Extracted {len(raw_text)} characters")
            
            chunks = await process_text(
                raw_text,
                max_words=self.max_words_per_chunk,
                overlap=self.chunk_overlap
            )
            
            if not chunks:
                raise ValueError("No chunks generated from text")
            
            logger.info(f"Generated {len(chunks)} chunks")
            
            chunk_texts = [chunk["text"] for chunk in chunks]
            lang_list = [chunk["language"] for chunk in chunks]
            
            embedding_results = embedding_service.embed(chunk_texts)
            
            vectors = []
            tokens_list = []
            for i in range(len(chunk_texts)):
                vectors.append(embedding_results[i]["embedding"])
                tokens_list.append(embedding_results[i]["tokens"])
            
            logger.info(f"Generated embeddings with dimension {len(vectors[0]) if vectors else 0}")
            
            upsert_result = pinecone_service.upsert(
                namespace=namespace,
                document_id=str(document_id),
                vectors=vectors,
                chunks=chunk_texts,
                lang_list=lang_list,
                tokens_list=tokens_list
            )
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            
            result = {
                "status": "success",
                "user_id": namespace,
                "document_id": str(document_id),
                "file_path": file_path,
                "num_chunks": len(chunks),
                "chunk_details": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "word_count": chunk["word_count"],
                        "language": chunk["language"]
                    }
                    for chunk in chunks
                ],
                "batches_upserted": upsert_result["batches"],
                "elapsed_time_seconds": round(elapsed_time, 2)
            }
            
            logger.info(f"Successfully ingested {document_id} in namespace {namespace} in {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest {document_id} in namespace {namespace}: {str(e)}")
            raise
    
    async def ingest_multiple_documents(
        self,
        namespace: str,
        file_paths: List[str],
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if document_ids and len(document_ids) != len(file_paths):
            raise ValueError("Number of document_ids must match number of file_paths")
        
        tasks = []
        for i, file_path in enumerate(file_paths):
            doc_id = document_ids[i] if document_ids else None
            task = self.ingest_document(file_path, doc_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "failed",
                    "namespace": namespace,
                    "file_path": file_paths[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        logger.info(f"Batch ingestion complete for namespace {namespace}: {successful}/{len(file_paths)} successful")
        
        return processed_results


pipeline = DocumentProcessingPipeline()
