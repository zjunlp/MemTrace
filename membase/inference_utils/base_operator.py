from __future__ import annotations
from .prompts import get_prompt 
from string import Template
import inspect
from abc import ABC, abstractmethod
from .backends import get_interface_for_inference
from tqdm import tqdm 
import time 
import os 
from typing import Any


class NonCachedLLMOperator(ABC): 
    """Base class for LLM operators that do not use caching."""

    def __init__(
        self, 
        prompt_name: str, 
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the operator.
        
        Args:
            prompt_name (`str`): 
                The name of the prompt template to use.
            model_name (`str | None`, optional): 
                The model name for the LLM inference backend. If not provided, the operator 
                is created without an inference interface and must be configured later.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to the inference backend.
        """
        params = inspect.signature(self._preprocess).parameters 
        for param in params:
            if not param.endswith("_list"):
                raise ValueError(
                    "The name of each argument in the `_preprocess` method must end with '_list'."
                )
        
        self.set_prompt(prompt_name)
        if model_name is not None:
            self._interface = get_interface_for_inference(model_name, **kwargs)
        else:
            self._interface = None

        self._model_name = model_name if model_name is not None else "Anonymous Model"
        # If the name of model is a directory path, 
        # we use the name of the directory as the model name. 
        if os.path.isdir(self._model_name):
            self._model_name = os.path.basename(self._model_name)

    def _check_prompt_identifiers(self) -> bool:
        """Check that prompt template identifiers match `_preprocess` arguments.
        
        Returns:
            `bool`: 
                Whether the prompt template identifiers match the arguments of the `_preprocess` function.
        """
        params = inspect.signature(self._preprocess).parameters
        return all(
            f"{identifier}_list" in params
            for identifier in self._prompt.get_identifiers()
        )

    @abstractmethod
    def _preprocess(self, *args) -> list[list[dict[str, Any]]]:
        """Convert input arguments into a list of message lists for the large language model.
        
        Subclasses should accept one or more parallel lists as arguments (each name must end 
        with `_list`), iterate over them, and use `self._prompt.substitute(...)` to fill the 
        prompt template for each item. For example, a question-answering operator might accept `question_list` 
        and an optional `context_list`, producing one message list per question.
        
        Returns:
            `list[list[dict[str, Any]]]`: 
                A list of message lists in OpenAI-style chat format.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _aggregate(self, responses: list[dict[str, Any]]) -> Any:
        """Aggregate the LLM responses into a final result.
        
        Subclasses can override this method to perform custom post-processing, such as 
        computing a metric from the responses or parsing structured output. The default 
        implementation returns the raw response list unchanged.
        
        Args:
            responses (`list[dict[str, Any]]`): 
                The list of LLM responses.
        
        Returns:
            `Any`: 
                The aggregated result.
        """
        return responses 
    
    @property
    def prompt(self) -> Template:
        """Return the prompt template."""
        return self._prompt

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def interface(self):
        """Return the inference interface."""
        return self._interface

    def _check(self) -> bool:
        """Return whether the inference interface is set."""
        return self._interface is not None
    
    def set_prompt(self, prompt: str | Template) -> None:
        """Set or update the prompt template. It will check if the prompt 
        identifiers match the arguments of the `_preprocess` function.
        
        Args:
            prompt (`str | Template`): 
                A prompt name or a `string.Template` instance.
        """
        if isinstance(prompt, Template):
            self._prompt = prompt
        else:
            self._prompt = get_prompt(prompt)
        if not self._check_prompt_identifiers():
            raise ValueError(
                "The prompt identifiers are not consistent with the arguments of the `_preprocess` function."
            )
    
    def from_operator(self, operator: NonCachedLLMOperator) -> None:
        """Copy the prompt, interface, and model name from another operator.
        
        Args:
            operator (`NonCachedLLMOperator`): 
                The source operator to copy from.
        """
        self.set_prompt(operator.prompt)
        self._interface = operator.interface
        self._model_name = operator.model_name
    
    def __call__(
        self, 
        *args: Any, 
        batch_size: int = 1, 
        aggregate: bool = True, 
        **kwargs: Any,
    ) -> Any:
        """Run the operator. It will preprocess inputs, call the LLM, and optionally aggregate results.
        
        Args:
            *args (`Any`): 
                Arguments forwarded to `_preprocess`.
            batch_size (`int`, defaults to `1`): 
                The number of messages to send per batch.
            aggregate (`bool`, defaults to `True`): 
                Whether to apply `_aggregate` on the collected responses.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to the inference interface.
        
        Returns:
            `Any`: 
                The aggregated result if `aggregate` is `True`, otherwise the raw list 
                of response dictionaries.
        """
        if not self._check():
            raise ValueError("The `interface` is not set.")
        
        messages_list = self._preprocess(*args)
        size = len(messages_list)

        progress_bar = tqdm(
            total=size, 
            desc=f"{self.__class__.__name__}(model={self._model_name})", 
        )
        final_responses = []
        
        for i in range(0, size, batch_size):
            batch_messages_list = [
                messages_list[batch_indice] 
                for batch_indice in range(i, min(i + batch_size, size))
            ]
            results = self._interface(batch_messages_list, **kwargs)
            if isinstance(results, dict):
                results = [results]
            for result in results:
                final_responses.append(result)
                progress_bar.update(1)
                # To make the progress bar update more smoothly.
                time.sleep(0.1)

        progress_bar.close()
        if aggregate:
            return self._aggregate(final_responses)
        return final_responses 