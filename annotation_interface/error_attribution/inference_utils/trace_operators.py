from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base_operator import NonCachedLLMOperator
from .operators import _parse_json_response


class StrictJSONModel(BaseModel):
    """Strict base model for parsing LLM JSON responses."""

    model_config = ConfigDict(extra="forbid", strict=True)


class TraceAnnotationCheckResult(StrictJSONModel):
    explanation: str = Field(...)
    is_annotation_error: bool = Field(...)
    bad_source_evidences: List[str] = Field(default_factory=list)


class TraceConstructionCheckResult(StrictJSONModel):
    explanation: str = Field(...)
    is_construction_error: bool = Field(...)
    op_id: Optional[str] = Field(default=None)


class TraceRetrievalCheckResult(StrictJSONModel):
    explanation: str = Field(...)
    is_retrieval_error: bool = Field(...)
    op_id: Optional[str] = Field(default=None)


class TraceResponseCheckResult(StrictJSONModel):
    explanation: str = Field(...)
    op_id: Optional[str] = Field(default=None)


class TraceErrorCaseAnalyzerResult(StrictJSONModel):
    summary: str = Field(...)
    suggestions: str = Field(...)


TRACE_RESPONSE_MODELS: Dict[str, type[StrictJSONModel]] = {
    "trace-annotation-check": TraceAnnotationCheckResult,
    "trace-construction-check": TraceConstructionCheckResult,
    "trace-retrieval-check": TraceRetrievalCheckResult,
    "trace-response-check": TraceResponseCheckResult,
}


def _parse_response_to_model(content: str, model_cls: type[StrictJSONModel]) -> StrictJSONModel:
    """Parse model output with a JSON fallback that handles markdown fences."""
    try:
        return model_cls.model_validate_json(content)
    except Exception:
        payload = _parse_json_response(content)
        return model_cls.model_validate(payload)


class TraceAnnotationChecker(NonCachedLLMOperator):
    """Judge whether source evidences are wrongly annotated for the query."""

    def __init__(self, prompt_name: str, model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(prompt_name=prompt_name, model_name=model_name, **kwargs)

    def _preprocess(
        self,
        question_list: List[str],
        golden_answer_list: List[List[str]],
        source_evidences_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for question, golden_answer, source_evidences in zip(
            question_list,
            golden_answer_list,
            source_evidences_list,
        ):
            if len(golden_answer) == 1:
                golden_answer_str = golden_answer[0]
            else:
                golden_answer_str = f"[{', '.join(golden_answer)}]"
            messages_list.append([
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question,
                        golden_answer=golden_answer_str,
                        source_evidences=source_evidences,
                    ),
                }
            ])
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_response_to_model(content, TraceAnnotationCheckResult)
                results.append(parsed.model_dump(mode="python"))
            except Exception:
                results.append(
                    TraceAnnotationCheckResult(
                        explanation=content,
                        is_annotation_error=False,
                        bad_source_evidences=[],
                    ).model_dump(mode="python")
                )
        return results


class TraceConstructionChecker(NonCachedLLMOperator):
    """Judge whether memory construction fails in the traced construction subgraph."""

    def __init__(self, prompt_name: str, model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(prompt_name=prompt_name, model_name=model_name, **kwargs)

    def _preprocess(
        self,
        question_list: List[str],
        golden_answer_list: List[List[str]],
        source_evidence_full_name_list: List[str],
        candidate_op_ids_list: List[str],
        construction_subgraph_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for question, golden_answer, source_name, subgraph, op_ids_text in zip(
            question_list,
            golden_answer_list,
            source_evidence_full_name_list,
            construction_subgraph_list,
            candidate_op_ids_list,
        ):
            if len(golden_answer) == 1:
                golden_answer_str = golden_answer[0]
            else:
                golden_answer_str = f"[{', '.join(golden_answer)}]"
            messages_list.append([
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question,
                        golden_answer=golden_answer_str,
                        source_evidence_full_name=source_name,
                        construction_subgraph=subgraph,
                        candidate_op_ids=op_ids_text,
                    ),
                }
            ])
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_response_to_model(content, TraceConstructionCheckResult)
                results.append(parsed.model_dump(mode="python"))
            except Exception:
                results.append(
                    TraceConstructionCheckResult(
                        explanation=content,
                        is_construction_error=False,
                        op_id=None,
                    ).model_dump(mode="python")
                )
        return results


class TraceRetrievalChecker(NonCachedLLMOperator):
    """Judge whether retrieval fails in the traced retrieval subgraph."""

    def __init__(self, prompt_name: str, model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(prompt_name=prompt_name, model_name=model_name, **kwargs)

    def _preprocess(
        self,
        question_list: List[str],
        golden_answer_list: List[List[str]],
        source_evidences_list: List[str],
        retrieval_subgraph_list: List[str],
        candidate_op_ids_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for question, golden_answer, source_evidences_text, subgraph_text, op_ids_text in zip(
            question_list,
            golden_answer_list,
            source_evidences_list,
            retrieval_subgraph_list,
            candidate_op_ids_list,
        ):
            if len(golden_answer) == 1:
                golden_answer_str = golden_answer[0]
            else:
                golden_answer_str = f"[{', '.join(golden_answer)}]"
            messages_list.append([
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question,
                        golden_answer=golden_answer_str,
                        source_evidences=source_evidences_text,
                        retrieval_subgraph=subgraph_text,
                        candidate_op_ids=op_ids_text,
                    ),
                }
            ])
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_response_to_model(content, TraceRetrievalCheckResult)
                results.append(parsed.model_dump(mode="python"))
            except Exception:
                results.append(
                    TraceRetrievalCheckResult(
                        explanation=content,
                        is_retrieval_error=False,
                        op_id=None,
                    ).model_dump(mode="python")
                )
        return results


class TraceResponseChecker(NonCachedLLMOperator):
    """Locate likely response-stage failure op_id from evaluation subgraph."""

    def __init__(self, prompt_name: str, model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(prompt_name=prompt_name, model_name=model_name, **kwargs)

    def _preprocess(
        self,
        question_list: List[str],
        golden_answer_list: List[List[str]],
        prediction_list: List[str],
        response_subgraph_list: List[str],
        candidate_op_ids_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for question, golden_answer, prediction, subgraph_text, op_ids_text in zip(
            question_list,
            golden_answer_list,
            prediction_list,
            response_subgraph_list,
            candidate_op_ids_list,
        ):
            if len(golden_answer) == 1:
                golden_answer_str = golden_answer[0]
            else:
                golden_answer_str = f"[{', '.join(golden_answer)}]"
            messages_list.append([
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question,
                        golden_answer=golden_answer_str,
                        prediction=prediction,
                        response_subgraph=subgraph_text,
                        candidate_op_ids=op_ids_text,
                    ),
                }
            ])
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_response_to_model(content, TraceResponseCheckResult)
                results.append(parsed.model_dump(mode="python"))
            except Exception:
                results.append(
                    TraceResponseCheckResult(
                        explanation=content,
                        op_id=None,
                    ).model_dump(mode="python")
                )
        return results


def get_trace_response_format(model_cls: type[StrictJSONModel]) -> Dict[str, Any]:
    """Build OpenAI json_schema response format directly from a Pydantic model class."""
    if not issubclass(model_cls, StrictJSONModel):
        raise ValueError(f"model_cls must inherit StrictJSONModel, got: {model_cls}")

    schema = model_cls.model_json_schema()
    # Some OpenAI-compatible backends require strict schemas where every key in
    # top-level `properties` must also appear in `required`.
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties.keys())

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": True,
            "schema": schema,
        },
    }