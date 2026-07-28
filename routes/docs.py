from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from logger.logger import get_logger
from database.database import db_manager
from features.doc_manage.doc__manage import DocumentRepository
from features.auth.dependencies import get_current_user
from features.doc_manage.schemas import (
    UserDocumentsResponse,
    DeleteDocumentResponse,
    DocumentResponse,
)

logger= get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
)


async def get_repository(
    db: AsyncSession = Depends(db_manager.get_session),
) -> DocumentRepository:
    return DocumentRepository(db)


@router.get("/", response_model=UserDocumentsResponse)
async def get_user_documents(
    current_user: dict = Depends(get_current_user),
    repository: DocumentRepository = Depends(get_repository),
):
    user_id = UUID(current_user["sub"])

    docs = await repository.get_user_documents(user_id)
    total = await repository.count_user_documents(user_id)

    return UserDocumentsResponse(
        user_id=user_id,
        documents=[DocumentResponse.model_validate(doc) for doc in docs],
        total=total,
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
)
async def delete_document(
    document_id: UUID,
    current_user: dict = Depends(get_current_user),
    repository: DocumentRepository = Depends(get_repository),
):
    user_id = UUID(current_user["sub"])

    document = await repository.delete_document(
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you don't have permission to delete it",
        )

    return DeleteDocumentResponse(
        status="success",
        message=f"Document '{document.file_name}' deleted successfully",
        document_id=document.id,
        file_name=document.file_name,
    )