from pydantic import BaseModel
from typing import List, Optional


class DocumentResponse(BaseModel):
    id: int
    file_name: str
    file_type: str
    file_size: int
    chunks: int
    primary_language: Optional[str]
    created_at: str
    updated_at: str


class DocumentsListResponse(BaseModel):
    user_id: int
    documents: List[DocumentResponse]