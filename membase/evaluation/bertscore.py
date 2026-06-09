from .base import BaseMetric 
from ..model_types.evaluation import MetricResult
from typing import ClassVar, Any


class BERTScore(BaseMetric):
    """BERTScore based on the `bert-score` package."""

    metric_name: ClassVar[str] = "bertscore"

    def __init__(
        self,
        lang: str = "en",
        model_type: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BERTScore metric.

        Args:
            lang (`str`, defaults to `"en"`):
                Language code. It is used to select the default model when
                the type of model is not provided.
            model_type (`str | None`, optional):
                HuggingFace model identifier (e.g., `"microsoft/deberta-xlarge-mnli"`). 
                When it is not provided, the default model for `lang` is used.
            batch_size (`int`, defaults to `64`):
                Batch size for the underlying transformer forward pass.
            device (`str | None`, optional):
                Device for the model (e.g., `"cuda:0"`). If it is not provided,
                it will be auto-detected.
        """
        self.lang = lang
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        try:
            from bert_score import score as bert_score_fn
        except ImportError as e:
            raise ImportError(
                "BERTScore metric requires the `bert-score` package. "
                "Install it with: pip install bert-score"
            ) from e

        score_kwargs = {
            "lang": self.lang,
            "verbose": False,
            "batch_size": self.batch_size,
        }
        if self.model_type is not None:
            score_kwargs["model_type"] = self.model_type
        if self.device is not None:
            score_kwargs["device"] = self.device

        P, R, F1 = bert_score_fn(
            cands=predictions,
            refs=references,
            **score_kwargs,
        )

        results = []
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
            results.append(
                {
                    "value": f,
                    "metadata": {
                        "precision": p,
                        "recall": r,
                    },
                }
            )
        return results
