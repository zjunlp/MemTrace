from ..evaluation import DEFAULT_METRICS, load_metrics
from ..model_types.evaluation import MetricResult
from ..model_types.dataset import MemoryDataset, QuestionAnswerPair
from typing import Any


class MemBaseDataset(MemoryDataset):
    """Intermediate base for all MemBase dataset implementations.

    It provides a default (empty) metadata generator, a JSON-based ``read_raw_data``
    loader that leverages Pydantic deserialization, and a ``save_dataset`` method
    for persisting the dataset to disk.  Dataset-specific subclasses should
    override ``read_raw_data`` to parse their own raw formats and may override
    ``_generate_metadata`` for richer statistics.
    """

    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        """Get the judge prompt template name for a question-answer pair.

        Subclasses can overwrite this method to customize the LLM-as-a-Judge prompt template.

        Args:
            qa_pair (`QuestionAnswerPair`):
                The question-answer pair to get the judge prompt template name for.

        Returns:
            `str`:
                The name of the judge prompt template.
        """
        return qa_pair.metadata.get("question_type", "default-exact-match")

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        """Convert the raw text output from the judge model into a correctness score.

        The default behaviour checks whether the word `"yes"` appears in the
        lowercased response. Subclasses can override this method to accommodate
        different judge formats.

        Args:
            content (`str`):
                The raw text content returned by the judge model.

        Returns:
            `float`:
                `1.0` if the prediction is judged correct, `0.0` otherwise.
        """
        return float("yes" in content.lower())

    @classmethod
    def evaluate(
        cls,
        qa_pairs: list[QuestionAnswerPair],
        predictions: list[str],
        metrics: list[str] | None = None,
        metric_configs: dict[str, dict[str, Any]] | None = None,
        judge_model: str = "gpt-4.1-mini",
        judge_batch_size: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, MetricResult]]:
        """Evaluate predictions against golden answers using configurable metrics.

        By default the metrics F1 score, BLEU score, and LLM-as-a-Judge score are computed.
        Users can select a different set via the ``metrics`` parameter and
        supply per-metric configuration through ``metric_configs``.

        Subclasses can also override this method to incorporate additional
        evaluation metrics beyond the built-in set.  For example, a dataset
        that annotates source evidence in each question-answer pair could
        compute retrieval recall@k by comparing the retrieved results against
        the ground-truth evidence IDs.  When overriding, call
        ``super().evaluate(...)`` first to obtain the base results, then merge
        the extra per-pair metrics into each result dictionary::

            @classmethod
            def evaluate(cls, qa_pairs, predictions, **kwargs):
                results = super().evaluate(qa_pairs, predictions, **kwargs)
                retrieval_results = kwargs.get("retrieval_results", [])
                for i, qa_pair in enumerate(qa_pairs):
                    evidence_ids = set(qa_pair.metadata.get("evidence_ids", []))
                    retrieved_ids = set(r["id"] for r in retrieval_results[i])
                    hit = len(evidence_ids & retrieved_ids)
                    k = len(retrieval_results[i]) if retrieval_results else 1
                    results[i][f"recall@{k}"] = {
                        "value": hit / len(evidence_ids) if evidence_ids else 0.0,
                        "metadata": {},
                    }
                return results

        Args:
            qa_pairs (`list[QuestionAnswerPair]`):
                The question-answer pairs to evaluate.
            predictions (`list[str]`):
                The predicted answers, one per question-answer pair.
            metrics (`list[str] | None`, optional):
                Names of metrics to compute. If it is not provided, the default metrics will be used.
            metric_configs (`dict[str, dict[str, Any]] | None`, optional):
                Per-metric keyword arguments keyed by metric name. For example,
                it can be `{"bleu": {"n_gram": 2, "lowercase": True}, "bertscore": {"lang": "zh"}}`.
            judge_model (`str`, defaults to `"gpt-4.1-mini"`):
                The model name or path used for the LLM judge.
            judge_batch_size (`int`, defaults to `4`):   
                Batch size for the judge model inference.
            **kwargs (`Any`):
                Remaining keyword arguments are forwarded to the LLM interface
                constructor.  If `api_key`, `api_keys`, `base_url` or
                `base_urls` is present, an OpenAI-compatible API backend is
                used.  Otherwise a local vLLM backend is assumed.  An optional
                `generation_config` dictionary can be included to supply
                generation-time parameters (mapping to the chat completions
                request body for OpenAI, or `vllm.SamplingParams` for vLLM). If `generation_config` is not provided,
                `{"temperature": 0.0}` is used by default for deterministic judging.

        Returns:
            `list[dict[str, MetricResult]]`:
                Per-pair evaluation results containing the metrics.
        """
        if len(qa_pairs) != len(predictions):
            raise ValueError(
                f"The number of question-answer pairs ({len(qa_pairs)}) and predictions "
                f"({len(predictions)}) must be the same."
            )

        metric_names = metrics if metrics is not None else DEFAULT_METRICS
        metric_configs = metric_configs or {}

        # Inject LLM judge-specific config from the top-level arguments so
        # that callers can keep using the familiar original signature 
        # without manually constructing metric_configs.
        if "llm_judge" in metric_names:
            judge_cfg = metric_configs.get("llm_judge", {})
            judge_cfg.setdefault("judge_model", judge_model)
            judge_cfg.setdefault("judge_batch_size", judge_batch_size)
            # Forward API-related kwargs to the LLM judge.
            for key in ("api_keys", "base_urls", "api_key", "base_url"):
                if key in kwargs and key not in judge_cfg:
                    judge_cfg[key] = kwargs[key]
            metric_configs["llm_judge"] = judge_cfg

        loaded_metrics = load_metrics(metric_names, metric_configs)

        # Build references list from question-answer pairs (multi-reference).
        references = [qa.golden_answers for qa in qa_pairs]

        results = [
            {} for _ in range(len(qa_pairs))
        ]

        for metric in loaded_metrics:
            # The LLM judge requires extra context beyond predictions and references.
            extra_kwargs = {}
            if metric.metric_name == "llm_judge":
                extra_kwargs["qa_pairs"] = qa_pairs
                extra_kwargs["get_judge_template_name"] = cls.get_judge_template_name
                extra_kwargs["parse_judge_response"] = cls.parse_judge_response
                if "generation_config" in kwargs:
                    extra_kwargs["generation_config"] = kwargs["generation_config"]

            metric_results = metric.compute(predictions, references, **extra_kwargs)
            for i, mr in enumerate(metric_results):
                results[i][metric.metric_name] = mr

        # Print the evaluation summary.
        cls.print_evaluation_summary(results, qa_pairs)

        return results

    @classmethod
    def print_evaluation_summary(
        cls,
        results: list[dict[str, dict[str, Any]]],
        qa_pairs: list[QuestionAnswerPair],
    ) -> None:
        """Print a summary of evaluation results grouped by metric and question type.

        This function prints the overall average value and a per-question-type breakdown
        (when more than one question type exists).

        Args:
            results (`list[dict[str, dict[str, Any]]]`):
                Per-pair evaluation results. Each element maps metric names to
                dictionaries containing at least a value key.
            qa_pairs (`list[QuestionAnswerPair]`):
                The corresponding question-answer pairs, used to extract
                question types from metadata.
        """
        if not results:
            return

        # Collect all metric names (preserving insertion order).
        metric_names = list(
            dict.fromkeys(
                key for result in results for key in result.keys()
            )
        )

        # Build question-type groups.
        question_type_groups = {}
        for idx, qa_pair in enumerate(qa_pairs):
            qtype = qa_pair.metadata.get("question_type", "unknown")
            question_type_groups.setdefault(qtype, []).append(idx)

        for metric in metric_names:
            values = [results[i][metric]["value"] for i in range(len(results))]
            overall = sum(values) / len(values)
            print(f"[{metric}] Overall: {overall:.4f}")

            if len(question_type_groups) > 1:
                for qtype, indices in sorted(question_type_groups.items()):
                    avg = sum(results[i][metric]["value"] for i in indices) / len(indices)
                    print(f"  {qtype}: {avg:.4f} ({len(indices)} questions)")
            print()
