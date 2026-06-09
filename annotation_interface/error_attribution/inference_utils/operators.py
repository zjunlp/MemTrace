from .base_operator import NonCachedLLMOperator
import numpy as np 
import json
import re
from typing import ( 
    List, 
    Dict, 
    Any, 
    Optional, 
)



class QuestionAnsweringOperator(NonCachedLLMOperator):
    """Operator that answers questions with an optional context."""

    def _preprocess(
        self, 
        question_list: list[str], 
        context_list: list[str] | None = None
    ) -> list[list[dict[str, str]]]: 
        """Build chat messages for each question.
        
        Args:
            question_list (`list[str]`): 
                The list of questions.
            context_list (`list[str] | None`, optional): 
                The list of contexts corresponding to each question. If provided, 
                the context is substituted into the prompt together with the question.
        
        Returns:
            `list[list[dict[str, str]]]`: 
                A list of OpenAI-style message lists.
        """
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            context = context_list[i] if context_list is not None else None
            if context is not None:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    }, 
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(question=question, context=context)
                    }, 
                ]
            else:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    }, 
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(question=question)
                    }, 
                ]
            messages_list.append(messages)
        return messages_list 


class LLMExactMatch(NonCachedLLMOperator):
    """Operator that uses an LLM to judge whether a prediction matches the golden answers."""

    def _preprocess(
        self, 
        question_list: list[str], 
        golden_answers_list: list[list[str]], 
        prediction_list: list[str], 
        reasoning_process_list: list[str] | None = None
    ) -> list[list[dict[str, str]]]: 
        """Build chat messages for each judgement request.
        
        When a question has multiple golden answers, they are formatted as a 
        bracketed comma-separated list in the prompt.
        
        Args:
            question_list (`list[str]`): 
                The list of questions.
            golden_answers_list (`list[list[str]]`): 
                The list of acceptable answer lists for each question.
            prediction_list (`list[str]`): 
                The list of model predictions.
            reasoning_process_list (`list[str] | None`, optional): 
                The list of reasoning processes. If provided, the reasoning process 
                is included in the prompt.
        
        Returns:
            `list[list[dict[str, str]]]`: 
                A list of OpenAI-style message lists.
        """
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            golden_answer_list = golden_answers_list[i]
            prediction = prediction_list[i]
            reasoning_process = reasoning_process_list[i] if reasoning_process_list is not None else None
            if len(golden_answer_list) == 1:
                golden_answer_list = golden_answer_list[0]
            else:
                golden_answer_list = f"[{', '.join(golden_answer for golden_answer in golden_answer_list)}]"
            if reasoning_process is None:
                messages = [
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(
                            question=question, 
                            golden_answers=golden_answer_list, 
                            prediction=prediction
                        )
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(
                            question=question, 
                            golden_answers=golden_answer_list,
                            reasoning_process=reasoning_process,
                            prediction=prediction
                        )
                    }
                ]
            messages_list.append(messages)
        return messages_list 


def _parse_json_response(content: str) -> Dict[str, Any]:
    """Parse a JSON response from an LLM output, handling markdown code blocks.

    If the content contains a fenced code block (with or without a ``json``
    language tag), the JSON is extracted from within the fence.  Otherwise
    the entire content string is parsed directly.

    Args:
        content (`str`):
            Raw string output from an LLM, potentially wrapped in a markdown
            code block.

    Returns:
        `Dict[str, Any]`:
            The parsed JSON object.

    Raises:
        `json.JSONDecodeError`:
            If the extracted string is not valid JSON.
    """
    # Try to extract JSON from a markdown code block first.
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()
    return json.loads(json_str)


class EvidenceAnnotationChecker(NonCachedLLMOperator):
    """Check whether a source evidence actually contains answer-relevant information.

    For each ``(question, golden_answers, source_evidence)`` triple, the
    operator prompts an LLM to decide whether the evidence is genuinely
    relevant to answering the question.  This step filters out annotation
    noise before downstream error attribution checks are performed.
    """

    def _preprocess(
        self,
        question_list: List[str],
        golden_answers_list: List[List[str]],
        source_evidence_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        """Build the per-sample message lists for the LLM.

        Args:
            question_list (`List[str]`):
                One question string per sample.
            golden_answers_list (`List[List[str]]`):
                One list of acceptable golden answers per sample.
            source_evidence_list (`List[str]`):
                One source evidence string per sample.

        Returns:
            `List[List[Dict[str, str]]]`:
                A list of single-turn message lists, one per sample, each
                formatted as ``[{"role": "user", "content": ...}]``.
        """
        messages_list = []
        for i in range(len(question_list)):
            golden_answers = golden_answers_list[i]
            if len(golden_answers) == 1:
                golden_answers_str = golden_answers[0]
            else:
                golden_answers_str = f"[{', '.join(golden_answers)}]"
            messages = [{
                "role": "user",
                "content": self._prompt.substitute(
                    question=question_list[i],
                    golden_answers=golden_answers_str,
                    source_evidence=source_evidence_list[i],
                )
            }]
            messages_list.append(messages)
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM responses into structured annotation check results.

        If JSON parsing fails for a response, the raw content is returned as
        the explanation and ``is_relevant`` defaults to ``True`` to avoid
        incorrectly discarding potentially valid evidences.

        Args:
            responses (`List[Dict[str, Any]]`):
                Raw response dictionaries from the LLM operator, each
                expected to contain a ``"processed_content"`` field.

        Returns:
            `List[Dict[str, Any]]`:
                A list of result dictionaries, one per response, each
                containing:

                - ``explanation`` (`str`): the model's reasoning.
                - ``is_relevant`` (`bool`): whether the evidence is relevant.
                - ``raw_response`` (`str`): the original response content.
        """
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append({
                    "explanation": parsed.get("explanation", ""),
                    "is_relevant": parsed.get("is_relevant", True),
                    "raw_response": content,
                })
            except (json.JSONDecodeError, KeyError):
                results.append({
                    "explanation": content,
                    "is_relevant": True,
                    "raw_response": content,
                })
        return results


class MemoryConstructionErrorChecker(NonCachedLLMOperator):
    """Check whether essential information from a source evidence is present in memory.

    For each ``(question, golden_answers, source_evidence, retrieved_memory_units)``
    tuple, the operator asks an LLM to determine whether the key information
    needed to answer the question has been successfully captured and is
    retrievable from the memory system.
    """

    def _preprocess(
        self,
        question_list: List[str],
        golden_answers_list: List[List[str]],
        source_evidence_list: List[str],
        retrieved_memory_units_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        """Build the per-sample message lists for the LLM.

        Args:
            question_list (`List[str]`):
                One question string per sample.
            golden_answers_list (`List[List[str]]`):
                One list of acceptable golden answers per sample.
            source_evidence_list (`List[str]`):
                One source evidence string per sample.
            retrieved_memory_units_list (`List[str]`):
                One pre-formatted string of retrieved memory units per sample,
                as produced by :func:`_build_memory_units_text`.

        Returns:
            `List[List[Dict[str, str]]]`:
                A list of single-turn message lists, one per sample, each
                formatted as ``[{"role": "user", "content": ...}]``.
        """
        messages_list = []
        for i in range(len(question_list)):
            golden_answers = golden_answers_list[i]
            if len(golden_answers) == 1:
                golden_answers_str = golden_answers[0]
            else:
                golden_answers_str = f"[{', '.join(golden_answers)}]"

            messages = [
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question_list[i],
                        golden_answers=golden_answers_str,
                        source_evidence=source_evidence_list[i],
                        retrieved_memory_units=retrieved_memory_units_list[i],
                    )
                }
            ]
            messages_list.append(messages)
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM responses into structured construction error check results.

        If JSON parsing fails for a response, the raw content is returned as
        the explanation and ``is_present`` defaults to ``False`` to conservatively
        flag the evidence as missing from memory.

        Args:
            responses (`List[Dict[str, Any]]`):
                Raw response dictionaries from the LLM operator, each
                expected to contain a ``"processed_content"`` field.

        Returns:
            `List[Dict[str, Any]]`:
                A list of result dictionaries, one per response, each
                containing:

                - ``explanation`` (`str`): the model's reasoning.
                - ``is_present`` (`bool`): whether the essential information
                  is present in the retrieved memory units.
                - ``raw_response`` (`str`): the original response content.
        """
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append(
                    {
                        "explanation": parsed.get("explanation", ""),
                        "is_present": parsed.get("is_present", False),
                        "raw_response": content,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Fallback: if JSON parsing fails, mark as not present.
                results.append(
                    {
                        "explanation": content,
                        "is_present": False,
                        "raw_response": content,
                    }
                )
        return results


class RootCauseAnalyzer(NonCachedLLMOperator):
    """Analyze the root cause of information loss during memory construction.

    Given the original message, the target evidence, and the sequence of LLM
    processing records produced while ingesting that message, the operator
    prompts an LLM to identify at which processing step the relevant
    information was dropped or distorted.
    """

    def _preprocess(
        self,
        evidence_list: List[str],
        raw_message_list: List[str],
        processing_records_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        """Build the per-sample message lists for the LLM.

        Args:
            evidence_list (`List[str]`):
                One target evidence string per sample — the information that
                should have been stored in memory.
            raw_message_list (`List[str]`):
                One JSON-serialized raw message string per sample, representing
                the original conversation turn that contained the evidence.
            processing_records_list (`List[str]`):
                One pre-formatted string of LLM processing records per sample,
                as produced by :func:`_format_processing_records`.

        Returns:
            `List[List[Dict[str, str]]]`:
                A list of single-turn message lists, one per sample, each
                formatted as ``[{"role": "user", "content": ...}]``.
        """
        messages_list = []
        for i in range(len(evidence_list)):
            messages = [
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        evidence=evidence_list[i],
                        raw_message=raw_message_list[i],
                        processing_records=processing_records_list[i],
                    )
                }
            ]
            messages_list.append(messages)
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[str]:
        """Extract the analysis text from each LLM response.

        Args:
            responses (`List[Dict[str, Any]]`):
                Raw response dictionaries from the LLM operator, each
                expected to contain a ``"processed_content"`` field.

        Returns:
            `List[str]`:
                A list of analysis strings, one per response.  An empty
                string is returned for any response that lacks
                ``"processed_content"``.
        """
        return [
            response.get("processed_content", "")
            for response in responses
        ]


class RetrievalErrorChecker(NonCachedLLMOperator):
    """Check whether retrieval results sufficiently cover the key source evidences.

    For each ``(question, golden_answers, source_evidences, retrieval_results)``
    tuple, the operator asks an LLM to determine whether the memory retrieval
    step has surfaced enough information to answer the question, given that
    the necessary evidence is known to be present in memory.
    """

    def _preprocess(
        self,
        question_list: List[str],
        golden_answers_list: List[List[str]],
        source_evidences_list: List[str],
        retrieval_results_list: List[str],
    ) -> List[List[Dict[str, str]]]:
        """Build the per-sample message lists for the LLM.

        Args:
            question_list (`List[str]`):
                One question string per sample.
            golden_answers_list (`List[List[str]]`):
                One list of acceptable golden answers per sample.
            source_evidences_list (`List[str]`):
                One pre-formatted string of source evidences per sample,
                as produced by :func:`_build_evidences_text`.
            retrieval_results_list (`List[str]`):
                One pre-formatted string of retrieved memory units per sample,
                as produced by :func:`_build_memory_units_text`.

        Returns:
            `List[List[Dict[str, str]]]`:
                A list of single-turn message lists, one per sample, each
                formatted as ``[{"role": "user", "content": ...}]``.
        """
        messages_list = []
        for i in range(len(question_list)):
            golden_answers = golden_answers_list[i]
            if len(golden_answers) == 1:
                golden_answers_str = golden_answers[0]
            else:
                golden_answers_str = f"[{', '.join(golden_answers)}]"

            messages = [
                {
                    "role": "user",
                    "content": self._prompt.substitute(
                        question=question_list[i],
                        golden_answers=golden_answers_str,
                        source_evidences=source_evidences_list[i],
                        retrieval_results=retrieval_results_list[i],
                    )
                }
            ]
            messages_list.append(messages)
        return messages_list

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM responses into structured retrieval error check results.

        If JSON parsing fails for a response, the raw content is returned as
        the explanation and ``is_sufficient`` defaults to ``False`` to
        conservatively flag the retrieval as insufficient.

        Args:
            responses (`List[Dict[str, Any]]`):
                Raw response dictionaries from the LLM operator, each
                expected to contain a ``"processed_content"`` field.

        Returns:
            `List[Dict[str, Any]]`:
                A list of result dictionaries, one per response, each
                containing:

                - ``explanation`` (`str`): the model's reasoning.
                - ``is_sufficient`` (`bool`): whether the retrieval results
                  sufficiently cover the source evidences.
                - ``raw_response`` (`str`): the original response content.
        """
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append(
                    {
                        "explanation": parsed.get("explanation", ""),
                        "is_sufficient": parsed.get("is_sufficient", False),
                        "raw_response": content,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Fallback: if JSON parsing fails, mark as not sufficient.
                results.append(
                    {
                        "explanation": content,
                        "is_sufficient": False,
                        "raw_response": content,
                    }
                )
        return results
    
class ErrorCaseAnalyzer(NonCachedLLMOperator):
    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append({
                    "summary": parsed.get("summary", ""),
                    "suggestions": parsed.get("suggestions", ""),
                    "raw_response": content,
                })
            except Exception:
                results.append({
                    "summary": content,
                    "suggestions": "",
                    "raw_response": content,
                })
        return results
    
class AnnotationErrorAnalyzer(ErrorCaseAnalyzer):
    """Analyzes common patterns across sampled annotation error cases."""

    def _preprocess(
        self,
        cases_text_list: List[str],
        num_cases_list: List[int],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for cases_text, num_cases in zip(cases_text_list, num_cases_list):
            messages_list.append([{
                "role": "user",
                "content": self._prompt.substitute(
                    cases_text=cases_text,
                    num_cases=num_cases,
                ),
            }])
        return messages_list


class ConstructionErrorAnalyzer(ErrorCaseAnalyzer):
    """Analyzes common patterns across sampled construction error cases."""

    def _preprocess(
        self,
        cases_text_list: List[str],
        num_cases_list: List[int],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for cases_text, num_cases in zip(cases_text_list, num_cases_list):
            messages_list.append([{
                "role": "user",
                "content": self._prompt.substitute(
                    cases_text=cases_text,
                    num_cases=num_cases,
                ),
            }])
        return messages_list


class RetrievalErrorAnalyzer(ErrorCaseAnalyzer):
    """Analyzes common patterns across sampled retrieval error cases."""

    def _preprocess(
        self,
        cases_text_list: List[str],
        num_cases_list: List[int],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for cases_text, num_cases in zip(cases_text_list, num_cases_list):
            messages_list.append([{
                "role": "user",
                "content": self._prompt.substitute(
                    cases_text=cases_text,
                    num_cases=num_cases,
                ),
            }])
        return messages_list


class ResponseErrorAnalyzer(ErrorCaseAnalyzer):
    """Analyzes common patterns across sampled response error cases."""

    def _preprocess(
        self,
        cases_text_list: List[str],
        num_cases_list: List[int],
    ) -> List[List[Dict[str, str]]]:
        messages_list = []
        for cases_text, num_cases in zip(cases_text_list, num_cases_list):
            messages_list.append([{
                "role": "user",
                "content": self._prompt.substitute(
                    cases_text=cases_text,
                    num_cases=num_cases,
                ),
            }])
        return messages_list