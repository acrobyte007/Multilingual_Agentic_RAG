from pydantic import BaseModel, Field, field_validator
from pathlib import Path

SUPPORTED_FILE_TYPES = {".pdf", ".doc", ".docx"}
MAX_FILE_NAME_LENGTH = 255
DEFAULT_LANGUAGE = "unknown"

class IngestResponse(BaseModel):
    status: str
    user_id: str
    document_id:int
    file_path: str
    num_chunks: int
    batches_upserted: int
    elapsed_time_seconds: float


class DocumentIngestRequest(BaseModel):
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_FILE_NAME_LENGTH,
    )

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        file_extension = Path(value).suffix.lower()

        if file_extension not in SUPPORTED_FILE_TYPES:
            raise ValueError(
                f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_FILE_TYPES))}"
            )

        return value