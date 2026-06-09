from .base import BaseMetric 
from ..model_types.evaluation import MetricResult
from typing import ClassVar, Any


class ROUGE(BaseMetric):
    """ROUGE score based on the `rouge-score` package."""

    metric_name: ClassVar[str] = "rouge"

    def __init__(
        self,
        rouge_types: list[str] | None = None,
        use_stemmer: bool = False,
    ) -> None:
        """Initialize the ROUGE metric.

        Args:
            rouge_types (`list[str] | None`, optional):
                ROUGE variants to compute. If it is not provided, the default variants 
                `["rouge1", "rouge2", "rougeL"]` will be used.
            use_stemmer (`bool`, defaults to `False`):
                Whether to apply Porter stemmer before scoring.
        """
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.use_stemmer = use_stemmer

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError(
                "ROUGE metric requires the `rouge-score` package. "
                "Install it with: pip install rouge-score"
            ) from e

        scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=self.use_stemmer,
        )

        # Determine which variant to use as the primary `value`.
        primary_key = "rougeL" if "rougeL" in self.rouge_types else self.rouge_types[-1]

        results = []
        for pred, refs in zip(predictions, references):
            best_scores = {}
            for ref in refs:
                scores = scorer.score(ref, pred)
                for key in self.rouge_types:
                    if key not in best_scores or scores[key].fmeasure > best_scores[key].fmeasure:
                        best_scores[key] = scores[key]

            metadata = {}
            for key in self.rouge_types:
                s = best_scores[key]
                metadata[key] = {
                    "precision": s.precision,
                    "recall": s.recall,
                    "fmeasure": s.fmeasure,
                }

            primary = best_scores[primary_key]
            value = primary.fmeasure

            results.append({"value": value, "metadata": metadata})
        return results
