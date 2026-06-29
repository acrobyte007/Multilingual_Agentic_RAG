from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from pathlib import Path
from features.retrieval.agent import get_rag_answer
from features.auth.dependencies import get_current_user
from fastapi import Depends
from features.ingestion.pipe_line import pipeline
from logger.logger import get_logger
from cache.conversation_save import cache

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


class IngestResponse(BaseModel):
    status: str
    user_id: str
    document_id: str
    file_path: str
    num_chunks: int
    batches_upserted: int
    elapsed_time_seconds: float


class QueryRequest(BaseModel):
    doc_ids: List[str]
    query: str
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["sub"])
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


@router.post("/response", response_model=QueryResponse)
async def get_rag_response(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["sub"])
    try:
        conversation_id = request.conversation_id

        if conversation_id is None:
            conversation_id = await cache.create_conversation(user_id)

        await cache.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.query,
            metadata={"namespace":user_id, "doc_ids": request.doc_ids}
        )

        conversation_history = await cache.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10,
            format_type="list"
        )

        response = await get_rag_answer(
            namespace=user_id,
            query=request.query,
            doc_ids=request.doc_ids,
            conversation=conversation_history
        )

        await cache.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response
        )

        return QueryResponse(
            answer=response,
            conversation_id=conversation_id
        )

    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))