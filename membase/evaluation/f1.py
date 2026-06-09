import re
import string
from collections import Counter
from .base import BaseMetric 
from ..model_types.evaluation import MetricResult
from typing import ClassVar, Any


def _normalize_answer(s: str) -> str:
    """It implements the SQuAD-standard normalization.
    
    Concretely, it lowers text and removes punctuation, articles and extra whitespace.

    Args:
        s (`str`): 
            The input string to be normalized.

    Returns:
        `str`: 
            The normalized string.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between a single prediction and reference.
    
    Args:
        prediction (`str`): 
            The prediction string.
        reference (`str`): 
            The reference string.

    Returns:
        `float`: 
            The token-level F1 score.
    """
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        # SQuAD v2 convention: 1.0 if both are empty (both "no answer"),
        # 0.0 if only one is empty.
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


class TokenF1(BaseMetric):
    """Token-level F1 score.

    For each example the F1 is computed between the prediction and every
    reference, and the maximum is taken.
    """

    metric_name: ClassVar[str] = "f1"

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        results = []
        for pred, refs in zip(predictions, references):
            f1 = max((_token_f1(pred, ref) for ref in refs), default=0.0)
            results.append({"value": f1, "metadata": {}})
        return results
