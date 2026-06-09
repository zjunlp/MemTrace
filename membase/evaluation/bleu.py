import re
from .base import BaseMetric 
from ..model_types.evaluation import MetricResult
from typing import ClassVar, Any


# Regex patterns for mteval-v13a tokenization used by WMT.
# Reference: https://github.com/huggingface/evaluate/blob/main/metrics/bleu/tokenizer_13a.py
_TOKENIZE_13a_REGEX = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def _tokenize_13a(text: str, lowercase: bool = False) -> list[str]:
    """Tokenize the text using mteval-v13a used by WMT.

    Args:
        text (`str`): 
            The input text to tokenize.
        lowercase (`bool`, defaults to `False`): 
            Whether to lowercase tokens.

    Returns:
        `list[str]`: 
            The token list.
    """
    line = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = (
            line.replace("&quot;", '"')
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )
    line = f" {line} "
    for _re, repl in _TOKENIZE_13a_REGEX:
        line = _re.sub(repl, line)
    tokens = line.split()
    return [t.lower() for t in tokens] if lowercase else tokens


class BLEU(BaseMetric):
    """Sentence-level BLEU score based on the `nltk` package."""

    metric_name: ClassVar[str] = "bleu"

    def __init__(
        self,
        n_gram: int = 1,
        smooth: bool = True,
        lowercase: bool = False,
    ) -> None:
        """Initialize the BLEU metric.

        Args:
            n_gram (`int`, defaults to `1`):
                Maximum n-gram order.
            smooth (`bool`, defaults to `True`):
                Whether to apply smoothing (method 1 of `nltk.translate.bleu_score.SmoothingFunction`).
            lowercase (`bool`, defaults to `False`):
                Whether to lowercase tokens before scoring. 
        """
        self.n_gram = n_gram
        self.smooth = smooth
        self.lowercase = lowercase

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError as e:
            raise ImportError(
                "BLEU metric requires the `nltk` package. "
                "Install it with: pip install nltk"
            ) from e

        weights = tuple(1.0 / self.n_gram for _ in range(self.n_gram))
        smooth_fn = SmoothingFunction().method1 if self.smooth else None

        results = []
        for pred, refs in zip(predictions, references):
            pred_tokens = _tokenize_13a(pred, lowercase=self.lowercase)
            ref_token_lists = [_tokenize_13a(ref, lowercase=self.lowercase) for ref in refs]
            try:
                score = sentence_bleu(
                    ref_token_lists,
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smooth_fn,
                )
            except (ZeroDivisionError, ValueError):
                score = 0.0
            results.append({"value": float(score), "metadata": {}})
        return results
