from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    doc_ids: List[str]
    query: str
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str