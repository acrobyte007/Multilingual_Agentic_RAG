import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ingestion.extraction import extract_text
from ingestion.cleaning_chunking import process_text
from ingestion.embedding_model import embedding_process
from database.vector_database import pinecone_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    
    def __init__(
        self,
        namespace: str,
        max_words_per_chunk: int = 150,
        chunk_overlap: int = 20,
        batch_size: int = 100
    ):
        self.namespace = namespace
        self.max_words_per_chunk = max_words_per_chunk
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        pinecone_service.batch_size = batch_size
    
    async def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        
        if document_id is None:
            document_id = Path(file_path).stem
        
        logger.info(f"Starting ingestion for namespace {self.namespace}, document: {document_id}")
        
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
            lang_list = [chunk["lang"] for chunk in chunks]
            
            embedding_results = embedding_process(chunk_texts)
            
            vectors = []
            tokens_list = []
            for i in range(len(chunk_texts)):
                vectors.append(embedding_results[i]["embedding"])
                tokens_list.append(embedding_results[i]["tokens"])
            
            logger.info(f"Generated embeddings with dimension {len(vectors[0]) if vectors else 0}")
            
            upsert_result = pinecone_service.upsert(
                namespace=self.namespace,
                document_id=document_id,
                vectors=vectors,
                chunks=chunk_texts,
                lang_list=lang_list,
                tokens_list=tokens_list
            )
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            
            result = {
                "status": "success",
                "namespace": self.namespace,
                "document_id": document_id,
                "file_path": file_path,
                "num_chunks": len(chunks),
                "batches_upserted": upsert_result["batches"],
                "elapsed_time_seconds": round(elapsed_time, 2)
            }
            
            logger.info(f"Successfully ingested {document_id} in namespace {self.namespace} in {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest {document_id} in namespace {self.namespace}: {str(e)}")
            raise
    
    async def ingest_multiple_documents(
        self,
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
                    "namespace": self.namespace,
                    "file_path": file_paths[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        logger.info(f"Batch ingestion complete for namespace {self.namespace}: {successful}/{len(file_paths)} successful")
        
        return processed_results


async def process_document(
    namespace: str,
    file_path: str,
    document_id: Optional[str] = None,
    max_words_per_chunk: int = 150,
    chunk_overlap: int = 20
) -> Dict[str, Any]:
    pipeline = DocumentProcessingPipeline(
        namespace=namespace,
        max_words_per_chunk=max_words_per_chunk,
        chunk_overlap=chunk_overlap
    )
    return await pipeline.ingest_document(file_path, document_id)