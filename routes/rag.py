from fastapi import APIRouter, HTTPException, UploadFile, File
import tempfile
import os
from pathlib import Path
from datetime import datetime
from services.agent import get_rag_answer
from features.auth.dependencies import get_current_user
from database.database_models import documents
from database.database import db_manager
from fastapi import Depends
from features.ingestion.pipe_line import pipeline
from features.ingestion.schema import IngestResponse, DocumentIngestRequest,DEFAULT_LANGUAGE
from features.retrieval.schema import QueryRequest, QueryResponse
from logger.logger import get_logger
from cache.conversation_save import cache

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["sub"])

    ingest_request = DocumentIngestRequest(file_name=file.filename)

    temp_file_path = None
    file_extension = Path(file.filename).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
        ) as temp_file:
            file_content = await file.read()
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        async with db_manager.connect() as session:
            document_record = documents(
                file_name=ingest_request.file_name,
                file_type=file_extension,
                file_size=os.path.getsize(temp_file_path),
                chunks=0,
                primary_language=DEFAULT_LANGUAGE,
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            session.add(document_record)
            await session.flush()

            ingestion_result = await pipeline.ingest_document(
                namespace=str(user_id),
                file_path=temp_file_path,
                document_id=document_record.id,
            )

            document_record.chunks = ingestion_result.get("num_chunks", 0)

            chunk_details = ingestion_result.get("chunk_details", [])
            if chunk_details:
                document_record.primary_language = chunk_details[0].get(
                    "language",
                    DEFAULT_LANGUAGE,
                )

            await session.commit()

        ingestion_result["document_id"] = document_record.id
        return IngestResponse(**ingestion_result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

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