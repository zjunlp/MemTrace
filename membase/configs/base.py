from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
import os
from typing import Self


class MemBaseConfig(BaseModel):
    """Base configuration for all memory layers."""

    user_id: str = Field(
        ..., 
        description="The user id of the memory system.",
    )
    save_dir: str = Field(
        default="memory", 
        description="The directory to save the memory.",
    )

    @model_validator(mode="after")
    def _validate_save_dir(self) -> Self:
        """Check the validity of the save directory."""
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"The provided path '{self.save_dir}' is a file. "
                "It should be a directory."
            )
        return self

    def get_llm_models(self) -> list[str]:
        """Get the large language models (LLMs) used in the memory layer.
        
        The default implementation returns an empty list. Subclasses should override this method to 
        list the LLM models whose token usage needs to be tracked during memory construction.

        Returns:
            `list[str]`: 
                A list containing the names of the LLM models used for memory construction.
        """
        return [] 