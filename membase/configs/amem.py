from .base import MemBaseConfig 
from pydantic import Field
from typing import Literal


class AMEMConfig(MemBaseConfig):
    """The default configuration for A-MEM."""

    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="The backend to use for the LLM. Currently, only openai and ollama are supported.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="The base backbone model to use.",
    )
    llm_api_key: str | None = Field(
        default=None,
        description=(
            "The API key to use for the LLM. It is used for openai backend. "
            "If not provided, the API key will be loaded from the environment variable."
        ),
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "The base URL to use for the LLM. It is used for openai backend. "
            "If not provided, the base URL will be loaded from the environment variable."
        ),
    )

    embedding_provider: Literal["sentence-transformers", "openai"] = Field(
        default="sentence-transformers",
        description="The provider for the embedding model.",
    )
    retriever_name_or_path: str = Field(
        default="all-MiniLM-L6-v2",
        description="The name or path of the retriever model to use.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description=(
            "The API key to use for the embedding model. It is used for openai backend. "
            "If not provided, the API key will be loaded from the environment variable."
        ),
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "The base URL to use for the embedding model. It is used for openai backend. "
            "If not provided, the base URL will be loaded from the environment variable."
        ),
    )

    # In A-MEM, each memory evolution operation modifies the keywords, tags, and context of notes. 
    # However, the corresponding embeddings are not updated. 
    # If the embeddings were updated every time a note is added, the overhead would be substantial. 
    # Therefore, A-MEM introduces a hyperparameter `evo_threshold`
    # where after adding `evo_threshold` notes, all note embeddings are updated.
    evo_threshold: int = Field(
        default=100,
        description="The threshold for the number of memories to trigger evolution.",
        gt=0,
    )

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]
