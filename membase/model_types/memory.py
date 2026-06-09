from pydantic import BaseModel, Field
from typing import Any


class MemoryEntry(BaseModel):
    """A single retrieved memory entry returned by all memory layers."""

    content: str = Field(
        description=(
            "The raw or primary semantic content of the memory entry "
            "as stored in the memory layer."
        ),
    )
    formatted_content: str | None = Field(
        default=None,
        description=(
            "A formatted representation of the memory entry, directly "
            "presented to the question-answering model as retrieved context."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Auxiliary information associated with the memory entry, such as "
            "timestamps, keywords, categories, or other backend-specific attributes."
        ),
    )