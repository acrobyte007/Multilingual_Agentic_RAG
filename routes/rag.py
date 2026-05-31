import os
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from llm_response.agent import get_rag_answer
from ingestion.pipe_line import DocumentProcessingPipeline

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

class IngestResponse(BaseModel):
    status: str
    namespace: str
    document_id: str
    file_path: str
    num_chunks: int
    batches_upserted: int
    elapsed_time_seconds: float

class PipelineManager:
    _pipelines: dict = {}
    
    @classmethod
    def get_pipeline(cls, namespace: str) -> DocumentProcessingPipeline:
        if namespace not in cls._pipelines:
            cls._pipelines[namespace] = DocumentProcessingPipeline(namespace=namespace)
        return cls._pipelines[namespace]

pipeline_manager = PipelineManager()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    namespace: str = Form(...),
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
        pipeline = pipeline_manager.get_pipeline(namespace)
        result = await pipeline.ingest_document(
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
    namespace: str = Form(...),
    user_query: str = Form(...),
    doc_ids: List[str] = Form(...)
):
    try:
        response = await get_rag_answer(namespace, user_query, doc_ids)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))