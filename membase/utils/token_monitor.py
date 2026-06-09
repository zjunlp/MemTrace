from __future__ import annotations
import functools
import json
from collections import deque
from threading import RLock, Lock
from litellm import token_counter as litellm_token_counter
from litellm.types.utils import SelectTokenizerResponse
from litellm import encoding
import tiktoken
from tokenizers import Tokenizer
from datetime import datetime
import inspect
from typing import (
    Any,
    ClassVar,
    Callable,
)


def get_tokenizer_for_model(model: str) -> SelectTokenizerResponse:
    """Get the tokenizer for a model.
    
    Args:
        model (`str`):
            The model name.
    
    Returns:
        `SelectTokenizerResponse`:
            The litellm tokenizer for the model.
    """
    try:
        tokenizer = Tokenizer.from_pretrained(model)
        return SelectTokenizerResponse(
            type="huggingface_tokenizer",
            tokenizer=tokenizer,
        )
    except Exception:
        print(
            f"Native huggingface tokenizer for {model} cannot be loaded. "
            "Load tiktoken tokenizer instead."
        )
        # See https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/token_counter.py#L504.
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            print(
                f"Tiktoken tokenizer for {model} cannot be loaded. "
                "Load litellm's default tokenizer instead."
            )
            tokenizer = encoding
        return SelectTokenizerResponse(
            type="openai_tokenizer",
            tokenizer=tokenizer,
        )


class CostState:
    """Cost state for a specific LLM model.
    
    It is used to track the cost of LLM API calls and provide statistics.
    """

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_time: float = 0.0,
        window_size: int = 100_000,
        total_count: int = 0,
        histories: list[dict[str, list[dict[str, str]] | str | float | int]] | None = None,
    ) -> None:
        """Initialize the cost state.
        
        Args:
            input_tokens (`int`, defaults to `0`):
                The cumulative number of input tokens.
            output_tokens (`int`, defaults to `0`):
                The cumulative number of output tokens.
            total_time (`float`, defaults to `0.0`):
                The cumulative time.
            window_size (`int`, defaults to `100_000`):
                The window size. It represents the number of calls to be stored in the history.
            total_count (`int`, defaults to `0`):
                The cumulative number of calls.
            histories (`list[dict[str, list[dict[str, str]] | str | float | int]]`, optional):
                The histories of the calls.
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time
        self.total_count = total_count
        self.histories = (
            deque(maxlen=window_size)
            if histories is None else
            deque(histories, maxlen=window_size)
        )
        self._lock = RLock()

    @property
    def total_tokens(self) -> int:
        """Compute the total number of tokens.
        
        Returns:
            `int`:
                The total number of tokens.
        """
        with self._lock:
            return self.input_tokens + self.output_tokens

    @property
    def average_input_tokens(self) -> float:
        """Compute the average number of input tokens per call.
        
        Returns:
            `float`:
                The average number of input tokens per call.
        """
        with self._lock:
            return self.input_tokens / max(self.total_count, 1)

    @property
    def average_output_tokens(self) -> float:
        """Compute the average number of output tokens per call.
        
        Returns:
            `float`:
                The average number of output tokens per call.
        """
        with self._lock:
            return self.output_tokens / max(self.total_count, 1)
        
    @property
    def average_tokens_per_call(self) -> float:
        """Compute the average number of tokens per call.
        
        Returns:
            `float`:
                The average number of tokens per call.
        """
        with self._lock:
            return self.total_tokens / max(self.total_count, 1)

    @property
    def average_time_per_call(self) -> float:
        """Compute the average time per call.
        
        Returns:
            `float`:
                The average time per call.
        """
        with self._lock:
            return self.total_time / max(self.total_count, 1)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the cost state to a dictionary.
        
        Returns:
            `dict[str, Any]`:
                The dictionary representation of the cost state.
        """
        with self._lock:
            return {
                "total_count": self.total_count,
                "total_tokens": self.total_tokens,
                "average_input_tokens": self.average_input_tokens,
                "average_output_tokens": self.average_output_tokens,
                "average_tokens_per_call": self.average_tokens_per_call,
                "average_time_per_call": self.average_time_per_call,
                "histories": list(self.histories),
                "total_time": self.total_time,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "window_size": self.histories.maxlen
            }

    def update(
        self,
        input_tokens: int,
        output_tokens: int,
        total_time: float,
        histories: list[dict[str, list[dict[str, str]] | str | float | int]],
    ) -> None:
        """Update the cost state.
        
        Args:
            input_tokens (`int`):
                The number of input tokens consumed by the corresponding historical calls.
            output_tokens (`int`):
                The number of output tokens consumed by the corresponding historical calls.
            total_time (`float`):
                The time consumed by the corresponding historical calls.
            histories (`list[dict[str, list[dict[str, str]] | str | float | int]]`):
                The new histories to be appended.
        """
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_time += total_time
            self.total_count += len(histories)
            self.histories.extend(histories)

    def to_json(self) -> str:
        """Convert the cost state to a JSON string.
        
        Returns:
            `str`:
                The JSON string representation of the cost state.
        """
        return json.dumps(
            self.to_dict(),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostState:
        """Create a cost state from a dictionary.
        
        Args:
            data (`dict[str, Any]`):
                The dictionary representation of the cost state.
                
        Returns:
            `CostState`:
                The cost state created from the dictionary.
        """
        allowed = [
            "input_tokens",
            "output_tokens",
            "total_time",
            "window_size",
            "total_count",
            "histories",
        ]
        kwargs = {k: data[k] for k in allowed if k in data}
        return cls(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> CostState:
        """Create a cost state from a JSON string.
        
        Args:
            json_str (`str`):
                The JSON string representation of the cost state.
                
        Returns:
            `CostState`:
                The cost state created from the JSON string.
        """
        return cls.from_dict(json.loads(json_str))


class CostStateManager:
    """Global manager for per-model CostState and tokenizers.

    This class cannot be instantiated. Use classmethods only.
    """
    _states: ClassVar[dict[str, CostState | dict[str, CostState]]] = {}
    _tokenizers: ClassVar[dict[str, SelectTokenizerResponse]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        raise OSError(
            "`CostStateManager` is designed to manage global cost states. "
            "It cannot be instantiated. Please use its classmethods instead."
        )

    @classmethod
    def register(
        cls,
        model: str,
        state: CostState | dict[str, CostState] | None = None,
        tokenizer: SelectTokenizerResponse | None = None,
        exist_ok: bool = False,
    ) -> None:
        """Register an existing cost state and optional tokenizer for a model.
        
        Args:
            model (`str`):
                The model name to register. It is used as the key for the cost state
                and the tokenizer.
            state (`CostState | dict[str, CostState] | None`, optional):
                An existing cost state to associate with the model. If it is not provided, a fresh
                cost state is created. It also supports a dictionary of operation type names to their
                respective cost states.
            tokenizer (`SelectTokenizerResponse | None`, optional):
                A pre-built tokenizer for the model. If it is not provided, one is automatically
                resolved based on the model name.
            exist_ok (`bool`, defaults to `False`):
                If it is enabled and the model is already registered, it will silently
                overwrite the existing entry. Otherwise, it will raise an error.
        """
        with cls._lock:
            if model in cls._states and not exist_ok:
                raise ValueError(
                    f"Model '{model}' has been already registered. Please pick another name."
                )
            # In the process of initialization, we can register a single model with a single `CostState`.
            # However, in the process of runtime, the number of `CostState` may be more than one.
            cls._states[model] = state or CostState()
            if tokenizer is not None:
                cls._tokenizers[model] = tokenizer
            else:
                cls._tokenizers[model] = get_tokenizer_for_model(model)

    @classmethod
    def get(cls, model: str) -> CostState | dict[str, CostState]:
        """Get the cost state for a registered model.
        
        Args:
            model (`str`):
                The model name whose cost state is to be retrieved.
        
        Returns:
            `CostState | dict[str, CostState]`:
                A single cost state if the model has no operation type distinction,
                or a dictionary mapping operation type names to their respective cost states.
        """
        with cls._lock:
            if model not in cls._states:
                raise KeyError(f"Model {model} is not registered. Please register it first.")
            return cls._states[model]

    @classmethod
    def update(
        cls,
        model: str,
        input_output_pair: dict[str, dict[str, list[dict[str, Any]] | str | float | int]],
        **kwargs
    ) -> None:
        """Update a model's cost state based on a given invocation record.
        
        An invocation record is represented as an input-output pair, which consists of an input
        dictionary, an output dictionary, and the elapsed wall-clock time. Both the input and
        output dictionaries must contain a ``"messages"`` key whose value is either a string or
        a list of messages in OpenAI-style chat format, representing the model's input and output respectively.
        These messages  are used to compute token consumption via the tokenizer registered for the model.

        You may also include additional metadata in the input and output dictionaries. In
        particular, this method checks whether the input dictionary contains a ``"metadata"`` key
        and whether that metadata contains an ``"op_type"`` field. If present, the token
        consumption for this model will be grouped by operation type. On the first such call, the
        single cost state is promoted to a dictionary mapping operation type names to their
        respective cost states. All subsequent calls must consistently include (or consistently
        omit) the operation type. This grouping enables more fine-grained analysis of token
        consumption across different operations.
        
        Args:
            model (`str`):
                The registered model name to update.
            input_output_pair (`dict[str, dict[str, list[dict[str, Any]] | str | float | int]]`):
                An invocation record composed of three keys "input", "output", and "elapsed".
                The `"input"` and `"output"` keys must contain a `"messages"` key whose value is either a
                string or a list of messages in OpenAI-style chat format. The `"elapsed"` key must be a
                float or an integer representing the elapsed wall-clock time in seconds.
            **kwargs:
                Additional keyword arguments forwarded to ``litellm.token_counter`` when computing
                input tokens (e.g., ``tools``, ``tool_choice``).
        """
        if "input" not in input_output_pair or "output" not in input_output_pair:
            raise ValueError("`input_output_pair` must contain both 'input' and 'output'.")
        if "elapsed" not in input_output_pair or not isinstance(input_output_pair["elapsed"], (int, float)):
            raise ValueError("'elapsed' must be provided as float seconds.")

        input_dict, output_dict = input_output_pair["input"], input_output_pair["output"]
        if "messages" not in input_dict or "messages" not in output_dict:
            raise ValueError("'input' and 'output' must contain 'messages'.")
        
        has_operation_type = "metadata" in input_dict and "op_type" in input_dict["metadata"]
        with cls._lock:
            cost_state = cls._states.get(model, None)
            if cost_state is None:
                raise KeyError(f"Model {model} is not registered. Please register it first.")
            tokenizer = cls._tokenizers.get(model)
            if has_operation_type:
                op_type = input_dict["metadata"]["op_type"]
                if isinstance(cost_state, CostState):
                    if len(cost_state.to_dict()["histories"]) > 0:
                        raise ValueError(
                            "Previous update operations do not contain an operation type. "
                            "However, the current update operation contains an operation type. "
                            "This is not allowed. Please make sure the `input_output_pair` is consistent "
                            "with the previous update operations."
                        )
                    else:
                        cls._states[model] = {}
                        cost_state = cls._states[model]
                if op_type not in cost_state:
                    cost_state[op_type] = CostState()
                cost_state = cost_state[op_type]
            elif isinstance(cost_state, dict):
                raise ValueError(
                    "Previous update operations contain different operation types "
                    "or the type of update operation has been inferred during registration. "
                    "However, the current update operation doesn't contain an operation type. "
                    "This is not allowed. Please make sure the `input_output_pair` is consistent "
                    "with the previous update operations."
                )

        inp = input_dict["messages"]
        out = output_dict["messages"]
        if not (isinstance(inp, (list, str)) and isinstance(out, (list, str))):
            raise TypeError("'messages' must be list[dict] or str for both input and output.")
        
        if isinstance(inp, list):
            input_tokens = litellm_token_counter(
                model=model,
                custom_tokenizer=tokenizer,
                messages=inp,
                **kwargs
            )
        else:
            input_tokens = litellm_token_counter(
                model=model,
                custom_tokenizer=tokenizer,
                text=inp,
                **kwargs
            )
        input_dict["input_tokens"] = input_tokens
        # NOTE: when we compute the output tokens, we don't consider
        # `tools`, `tool_choice`, `use_default_image_token_count`, and `default_token_count`
        # as they are taken into account when we compute the input tokens.
        if isinstance(out, list):
            output_tokens = litellm_token_counter(
                model=model,
                custom_tokenizer=tokenizer,
                messages=out,
            )
        else:
            output_tokens = litellm_token_counter(
                model=model,
                custom_tokenizer=tokenizer,
                text=out,
            )
        output_dict["output_tokens"] = output_tokens

        # Update the corresponding cost state.
        cost_state.update(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_time=input_output_pair["elapsed"],
            histories=[input_output_pair],
        )

    @classmethod
    def reset(cls) -> None:
        """Reset all registered models by clearing every cost state and tokenizer."""
        with cls._lock:
            cls._states.clear()
            cls._tokenizers.clear()
    
    @classmethod
    def save_to_json_file(cls, filename: str) -> None:
        """Save all registered cost states to a JSON file.
        
        Args:
            filename (`str`):
                The base filename (without the `.json` extension).
        """
        with cls._lock:
            output_dict = {}
            for model, state in cls._states.items():
                if isinstance(state, CostState):
                    output_dict[model] = state.to_dict()
                else:
                    output_dict[model] = {op: cs.to_dict() for op, cs in state.items()}
            with open(f"{filename}.json", 'w') as f:
                json.dump(
                    output_dict,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    sort_keys=True,
                )


def token_monitor(
    extract_model_name: Callable[..., tuple[str, dict[str, Any]]],
    extract_input_dict: Callable[..., dict[str, list[dict[str, Any]] | str | float | int]],
    extract_output_dict: Callable[..., dict[str, list[dict[str, Any]] | str | float | int]],
) -> Callable:
    """
    Decorator to monitor token usage and latency for large language model (LLM) API calls.
    
    This decorator wraps sync or async callables, extracts model name and I/O payloads,
    computes input/output tokens, measures elapsed time, and appends a structured record
    to the per-model cost state managed by the global cost state manager. The target function must
    complete successfully for an update to be recorded.
    
    Args:
        extract_model_name (`Callable[..., tuple[str, dict[str, Any]]]`):
            A callable that returns a tuple containing the model name and metadata. The model name
            is an identifier used to retrieve the corresponding cost state from the global cost state manager.
            The extracted metadata is used to configure the LiteLLM's token counter.
        extract_input_dict (`Callable[..., dict[str, list[dict[str, Any]] | str | float | int]]`):
            A callable that builds the input dictionary. It must include a `"messages"` key
            whose value is either in OpenAI-style chat format or a string. A `"timestamp"` string will be
            injected by the decorator. Users can customize this callable to extract related metadata to
            conduct further analysis on the token usage.
        extract_output_dict (`Callable[..., dict[str, list[dict[str, Any]] | str | float | int]]`):
            A callable that builds the output dictionary from the function result. It must include
            a `"messages"` key. A `"timestamp"` string will be injected by the decorator. Users can customize
            this callable to extract related metadata to conduct further analysis on the token usage.
        
    Returns:
        `Callable`:
            A wrapper that preserves the original LLM call behavior (i.e., it executes the
            wrapped function normally) and, on top of that, records the elapsed time and
            token usage. The wrapper preserves the original function's signature and
            supports both synchronous and asynchronous callables.
    
    Notes:
        - Before using the decorator, register the model via
        `CostStateManager.register(model, state=..., tokenizer=...)`. Otherwise, an update
        will raise `KeyError`.
        - The record (i.e., an input-output pair) pushed to the cost state has the following schema:

            ```json
            {
                "input": <input_dict>,
                "output": <output_dict>,
                "elapsed": <float seconds>,
                "function_name": <str>,
                "is_success": <bool>,
                "err_msg": <str | None>,
            }
            ```

        - Token counting is performed by LiteLLM's `token_counter` and can leverage a
        `custom_tokenizer` provided through the returned `metadata` from `extract_model_name`.
        - For async functions, the wrapper awaits the coroutine and then performs accounting.
    
    Examples:
        Synchronous usage::
            
            @token_monitor(
                extract_model_name=lambda *args, **kwargs: ("gpt-4o-mini", {"custom_tokenizer": None}),
                extract_input_dict=lambda *args, **kwargs: {"messages": kwargs["messages"]},
                extract_output_dict=lambda result: {"messages": result["messages"]},
            )
            def call_llm(messages):
                ...
            
        Asynchronous usage::
            
            @token_monitor(
                extract_model_name=lambda *args, **kwargs: ("claude-3-sonnet", {}),
                extract_input_dict=lambda *args, **kwargs: {"messages": kwargs["messages"]},
                extract_output_dict=lambda result: {"messages": result["messages"]},
            )
            async def a_call_llm(messages):
                ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                model_name, metadata = extract_model_name(*args, **kwargs)
                input_dict = extract_input_dict(*args, **kwargs)
                start_time = datetime.now().astimezone()
                input_dict["timestamp"] = start_time.isoformat(timespec="seconds")
                err = None
                result = None
                has_result = False
                try:
                    result = await func(*args, **kwargs)
                    has_result = True
                except Exception as e:
                    err = e
                    print(f"Error in {func.__name__}: \n\t{e.__class__.__name__}: {e}")
                finally:
                    end_time = datetime.now().astimezone()
                    if has_result:
                        output_dict = extract_output_dict(result)
                    else:
                        output_dict = {"messages": ""}
                    output_dict["timestamp"] = end_time.isoformat(timespec="seconds")
                    CostStateManager.update(
                        model_name,
                        {
                            "input": input_dict,
                            "output": output_dict,
                            "elapsed": (end_time - start_time).total_seconds(),
                            "function_name": func.__name__,
                            "is_success": has_result,
                            "err_msg": (
                                None if err is None
                                else f"{err.__class__.__name__}: {err}"
                            ),
                        },
                        **metadata,
                    )
                if err is not None:
                    raise RuntimeError(
                        "The decorated function failed during execution. "
                        "Please check the printed error above, or inspect the token cost file "
                        "for the error message."
                    ) from err
                return result
            return awrapper
        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract the model name and metadata used during the token computation
                # The extraction function should be provided by the user
                model_name, metadata = extract_model_name(*args, **kwargs)

                # Extract the input dictionary
                input_dict = extract_input_dict(*args, **kwargs)
                start_time = datetime.now().astimezone()
                input_dict["timestamp"] = start_time.isoformat(timespec="seconds")

                err = None
                result = None
                has_result = False
                try:
                    # Run the original function.
                    result = func(*args, **kwargs)
                    has_result = True
                except Exception as e:
                    err = e
                    print(f"Error in {func.__name__}: \n\t{e.__class__.__name__}: {e}")
                finally:
                    end_time = datetime.now().astimezone()
                    if has_result:
                        # Extract the output dictionary.
                        output_dict = extract_output_dict(result)
                    else:
                        output_dict = {"messages": ""}
                    output_dict["timestamp"] = end_time.isoformat(timespec="seconds")
                    # Update the cost state.
                    CostStateManager.update(
                        model_name,
                        {
                            "input": input_dict,
                            "output": output_dict,
                            "elapsed": (end_time - start_time).total_seconds(),
                            "function_name": func.__name__,
                            "is_success": has_result,
                            "err_msg": (
                                None if err is None
                                else f"{err.__class__.__name__}: {err}"
                            ),
                        },
                        **metadata,
                    )
                if err is not None:
                    raise RuntimeError(
                        "The decorated function failed during execution. "
                        "Please check the printed error above, or inspect the token cost file "
                        "for the error message."
                    ) from err
                return result
            return wrapper
        
    return decorator
