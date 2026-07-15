import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from datetime import datetime
from features.doc_manage.schemas import DocumentsListResponse
from database.database import db_manager
from database.database_models import documents
from features.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.get("/", response_model=DocumentsListResponse)
async def get_user_documents(
    current_user: dict = Depends(get_current_user)
):
    user_id = uuid.UUID(current_user["sub"])

    try:
        async with db_manager.connect() as session:

            stmt = select(documents).where(documents.user_id == user_id)
            result = await session.execute(stmt)
            docs = result.scalars().all()

            response_docs = [
                {
                    "id": doc.id,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "chunks": doc.chunks,
                    "primary_language": doc.primary_language,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                }
                for doc in docs
            ]

            return {
                "user_id": user_id,
                "documents": response_docs
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))