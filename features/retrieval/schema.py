from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Annotated
from pydantic import StringConstraints

class QueryRequest(BaseModel):
    doc_ids: List[str]
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str