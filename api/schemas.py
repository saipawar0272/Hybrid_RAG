
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User question for the RAG application"
    )

    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Conversation identifier"
    )

    @field_validator("question", "thread_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Value cannot contain only whitespace")

        return value


class QueryResponse(BaseModel):
    question: str
    answer: str
    thread_id: str