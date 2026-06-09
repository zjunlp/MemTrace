from .dataset import Message
from typing import TypedDict, Any


class MetricResult(TypedDict):
    """Typed result for a single example from a single metric."""

    value: float
    """The scalar score for this example."""

    metadata: dict[str, Any]
    """Auxiliary information (e.g., per-sub-metric breakdown)."""


class OnlineEvalResult(TypedDict):
    """Result of a single online evaluation episode."""

    metrics: dict[str, MetricResult]
    """Per-metric evaluation scores."""

    rollout: list[Message]
    """The full interaction trace for this evaluation episode."""
