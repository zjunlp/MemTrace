import inspect
from smartcomment import (
    comment_variable, 
    comment_op,
    comment_op_scope,
    comment_link, 
)
from .base import BaseMetric, MetricResult
from typing import ClassVar, Any


class LLMJudge(BaseMetric):
    """LLM-as-a-Judge metric based on `vllm` or `openai`."""

    metric_name: ClassVar[str] = "llm_judge"

    def __init__(
        self,
        judge_model: str = "gpt-4.1-mini",
        judge_batch_size: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLMJudge metric.

        Args:
            judge_model (`str`, defaults to `"gpt-4.1-mini"`):
                Model name or path for the judge LLM.
            judge_batch_size (`int`, defaults to `4`):
                Batch size for judge inference.
            **kwargs (`Any`):
                Extra keyword arguments forwarded to the LLM interface
                constructor.
        """
        self.judge_model = judge_model
        self.judge_batch_size = judge_batch_size
        self.interface_kwargs = kwargs

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """Run the LLM judge.

        In addition to the standard arguments, the caller must provide:

        - qa_pairs (`list[QuestionAnswerPair]`): 
            The question-answer pairs, used to build judge prompts.
        - get_judge_template_name (`Callable[[QuestionAnswerPair], str]`): 
            It resolves a question-answer pair to a prompt template name.
        - parse_judge_response (`Callable[[str], float]`): 
            It converts raw judge text to a float score.

        An optional `generation_config` dictionary can be included.

        Args:
            predictions (`list[str]`):
                Model outputs. It is unused directly.
            references (`list[list[str]]`):
                Golden answers. It is unused directly.
            **kwargs (`Any`):
                It must contain keys `qa_pairs`, `get_judge_template_name`,
                and `parse_judge_response`.

        Returns:
            `list[MetricResult]`:
                Per-example judge scores.
        """
        from ..inference_utils.operators import LLMExactMatch

        qa_pairs = kwargs["qa_pairs"]
        get_judge_template_name = kwargs["get_judge_template_name"]
        parse_judge_response = kwargs["parse_judge_response"]
        generation_config = {"temperature": 0.0}
        generation_config.update(kwargs.get("generation_config", {}))

        # Group question-answer pairs by judge template name for batched evaluation.
        groups = {}
        for idx, qa_pair in enumerate(qa_pairs):
            template_name = get_judge_template_name(qa_pair)
            groups.setdefault(template_name, []).append(idx)

        results = [
            {"value": 0.0, "metadata": {}} for _ in range(len(qa_pairs))
        ]

        first_template = next(iter(groups))
        judge_operator = LLMExactMatch(
            prompt_name=first_template,
            model_name=self.judge_model,
            **self.interface_kwargs,
        )

        for template_name, indices in groups.items():
            judge_operator.set_prompt(template_name)

            runtime_judge_template = comment_variable(
                judge_operator.prompt.template,
                to_runtime=True,
                id_strategy=lambda _: template_name,
                comment=(
                    "The prompt template for the judge model. "
                    "It is a `string.Template` object with `$question`, `$prediction`, "
                    "and `$golden_answers` placeholders. "
                    "It tells the judge model to judge whether the model's prediction " 
                    "is correct or not based on a list of golden answers."
                ),
                category="prompt",
                class_name="judge_prompt",
                metadata={
                    "op_type": "llm-judge",
                }
            )

            batch_questions = [qa_pairs[i].question for i in indices]
            batch_golden_answers = [qa_pairs[i].golden_answers for i in indices]
            batch_predictions = [predictions[i] for i in indices]

            judge_responses = judge_operator(
                batch_questions,
                batch_golden_answers,
                batch_predictions,
                batch_size=self.judge_batch_size,
                aggregate=False,
                **generation_config,
            )

            for local_pos, global_idx in enumerate(indices):
                content = judge_responses[local_pos].get("processed_content")
                if content is None:
                    raise ValueError(
                        "The judge model's response for question "
                        f"'{qa_pairs[global_idx].question}' is empty."
                    )
                results[global_idx] = {
                    "value": parse_judge_response(content),
                    "metadata": {"judge_response": content},
                }

                with comment_op_scope(
                    op_name="llm-judge",
                    category="evaluation",
                    comment=(
                        "The judge model judges whether the model's prediction "
                        "is correct or not based on a list of golden answers."
                    ),
                ):
                    question = (
                        batch_questions[local_pos], 
                        {
                            "class_name": "query", 
                            "id_strategy": lambda v: qa_pairs[global_idx].id,
                        }
                    )
                    prediction = (
                        batch_predictions[local_pos], 
                        {
                            "class_name": "prediction",
                            "category": "llm_response",
                        }
                    )
                    golden_answers = (
                        batch_golden_answers[local_pos], 
                        {
                            "class_name": "golden_answers", 
                            "encoding_fn": lambda v: ", ".join(v),
                            "category": "golden_answers",
                            "comment": "The golden answers for the question.",
                        }
                    )
                    judge_response = (
                        content, 
                        {
                            "class_name": "judge_response",
                            "category": "llm_response",
                            "comment": "The judge model's judgment.",
                        }
                    )

                    comment_op(
                        inputs=[
                            question, 
                            golden_answers, 
                            runtime_judge_template,
                            prediction,
                        ],
                        outputs=[judge_response],
                        comment=(
                            "The judge model gives its judgment " 
                            "based on the question, the golden answers, " 
                            "and the instruction."
                        ),
                        reuse_op=True,
                    )
                    comment_link(
                        source=judge_response,
                        target=(
                            results[global_idx]["value"],
                            {
                                "class_name": "judge_score", 
                                "comment": "The judgement score.",
                            },
                        ),
                        comment=(
                            "Use a function to convert the judge model's response "
                            "to a float score."
                        ),
                        edge_metadata={
                            "source_code": inspect.getsource(parse_judge_response),
                        },
                    )
                    

        return results
