import os
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from llm_response.agent import get_rag_answer
from ingestion.pipe_line import pipeline
from logger.logger import get_logger
from cache.conversation import cache
logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


class IngestResponse(BaseModel):
    status: str
    namespace: str
    document_id: str
    file_path: str
    num_chunks: int
    batches_upserted: int
    elapsed_time_seconds: float


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    supported_extensions = {".pdf", ".docx", ".doc"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
        )
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        result = await pipeline.ingest_document(
            namespace=user_id,
            file_path=tmp_path,
            document_id=document_id
        )
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/api/v1/rag/response")
async def get_rag_response(
    user_query: str = Form(...),
    doc_ids: List[str] = Form(...),
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
):
    try:
        if conversation_id is None:
            conversation_id = await cache.create_conversation(user_id)
        
        await cache.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_query,
            metadata={"namespace": user_id, "doc_ids": doc_ids}
        )
        
        conversation_history = await cache.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10,
            format_type="list"
        )
        
        response = await get_rag_answer(
            namespace=user_id,
            query=user_query,
            doc_ids=doc_ids,
            conversation=conversation_history
        )
        
        await cache.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response
        )
        
        return {
            "response": response,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))