from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping
from .base import BaseMetric
from typing import Any


_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ("f1", "TokenF1"),
        ("bleu", "BLEU"),
        ("rouge", "ROUGE"),
        ("bertscore", "BERTScore"),
        ("llm_judge", "LLMJudge"),
    ]
)

_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("f1", "f1"),
        ("bleu", "bleu"),
        ("rouge", "rouge"),
        ("bertscore", "bertscore"),
        ("llm_judge", "llm_judge"),
    ]
)

METRIC_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)


DEFAULT_METRICS = ["f1", "bleu", "llm_judge"]

def load_metrics(
    metric_names: list[str] | None = None,
    metric_configs: dict[str, dict[str, Any]] | None = None,
) -> list[BaseMetric]:
    """Instantiate metrics by name with optional per-metric configuration.

    Args:
        metric_names (`list[str] | None`, optional):
            Names of metrics to load. If it is not provided, the default metrics
            will be used.
        metric_configs (`dict[str, dict[str, Any]] | None`, optional):
            Per-metric keyword arguments keyed by metric name. For example,
            `{"bleu": {"n_gram": 2, "lowercase": True}, "bertscore": {"lang": "zh"}}`.

    Returns:
        `list[BaseMetric]`:
            Instantiated metric objects.
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS
    if metric_configs is None:
        metric_configs = {}

    metrics = []
    for name in metric_names:
        cls = METRIC_MAPPING[name]
        config = metric_configs.get(name, {})
        metrics.append(cls(**config))
    return metrics


__all__ = [
    "METRIC_MAPPING",
    "DEFAULT_METRICS",
    "load_metrics",
]
