import os
import json
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from typing import Self


class _ApiPool(BaseModel):
    """OpenAI-compatible API credential pool for concurrent agents."""

    api_keys: list[str] = Field(
        description="API keys for the agents.",
    )
    base_urls: list[str] = Field(
        description="Base URLs for the agents.",
    )

    @property
    def size(self) -> int:
        """Return the number of available credential slots.

        Returns:
            `int`:
                The number of available credential slots.
        """
        return len(self.api_keys)

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        """Validate the credential slot layout.

        Returns:
            `Self`:
                The validated API credential pool.
        """
        if len(self.api_keys) == 0:
            raise ValueError("At least one API key slot is required.")
        if len(self.api_keys) != len(self.base_urls):
            raise ValueError("`api_keys` and `base_urls` must have the same length.")
        return self

    def credential_for(self, index: int) -> tuple[str, str]:
        """Return one credential pair by round-robin index.

        Args:
            index (`int`):
                The index.

        Returns:
            `tuple[str, str]`:
                A tuple containing the API key and base URL.
        """
        slot = index % len(self.api_keys)
        return self.api_keys[slot], self.base_urls[slot]


class AgentBaseConfig(BaseModel):
    """A configuration for the base agent runner."""

    model: str = Field(
        default="gpt-4.1",
        description="The backbone model used for the agent.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        description="Sampling temperature for the agent.",
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream agent responses.",
    )
    studio_url: str | None = Field(
        default=None,
        description=(
            "The URL of the AgentScope Studio server. "
            "If the effective batch size is greater than 1, "
            "the provided URL will be ignored."
        ),
    )
    project: str = Field(
        default="optimization",
        description="The project name in the AgentScope Studio.",
    )
    api_config_path: str | None = Field(
        default=None,
        description="Path to the API config file.",
    )
    api_keys: list[str] | None = Field(
        default=None,
        description=(
            "API keys for the agent. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    base_urls: list[str] | None = Field(
        default=None,
        description=(
            "Base URLs for the agent. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )


class AgentBaseRunner:
    """Runner that concurrently runs agents."""

    def __init__(self, config: AgentBaseConfig | None = None) -> None:
        """Initialize the base agent runner.

        Args:
            config (`AgentBaseConfig | None`, optional):
                The base agent runner configuration. If not provided, 
                default configuration is used.
        """
        self.config = config or AgentBaseConfig()
        self._initialize_api_pool()

    def _initialize_api_pool(self) -> None:
        """Initialize API credentials from config, file, or environment."""
        cfg = self.config
        if cfg.api_keys is not None and cfg.base_urls is not None:
            self._api_pool = _ApiPool(
                api_keys=cfg.api_keys,
                base_urls=cfg.base_urls,
            )
            return 

        if cfg.api_keys is not None or cfg.base_urls is not None:
            raise ValueError(
                "Either both `api_keys` and `base_urls` must be provided, "
                "or neither."
            )

        if cfg.api_config_path is not None:
            with open(cfg.api_config_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            self._api_pool = _ApiPool(
                api_keys=payload.get("api_keys", []),
                base_urls=payload.get("base_urls", []),
            )
            return 

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        if base_url is None:
            raise ValueError("No API credentials are found.") 
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._api_pool = _ApiPool(
            api_keys=[api_key],
            base_urls=[base_url],
        )