from .base_operator import NonCachedLLMOperator


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

