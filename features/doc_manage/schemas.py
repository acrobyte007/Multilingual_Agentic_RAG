import uuid
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    file_name: str
    file_type: str
    file_size: int
    filebase_key: str | None = None
    filebase_url: str | None = None
    bucket_name: str
    chunks: int
    primary_language: str
    created_at: datetime
    updated_at: datetime


class UserDocumentsResponse(BaseModel):
    user_id: UUID
    documents: List[DocumentResponse]
    total: int


class DeleteDocumentResponse(BaseModel):
    status: str
    message: str
    document_id: UUID
    file_name: str