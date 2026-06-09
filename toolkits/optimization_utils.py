import asyncio
import random
import re 
import os 
import json 
import time
import warnings
from pathlib import Path
from collections import deque, defaultdict 
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.agent import ReActAgent
from graphtrace import (
    ChatUsageTokenMonitor,
    StudioServer,
    agentscope_token_monitor,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    Field,
    ModelWrapValidatorHandler,
    model_validator,
    PrivateAttr,
    SerializeAsAny,
)
from membase.configs import CONFIG_MAPPING
from membase.datasets import DATASET_MAPPING
from membase.model_types.dataset import MemoryDataset 
from membase.configs.base import MemBaseConfig 
from smartcomment.runtime import ExecNetwork 
from ._base import AgentBaseConfig, AgentBaseRunner
from .memtrace_utils import ErrorAttributionPrediction
from typing import (
    Any, 
    Self,
    Literal,
)


ITERATION_PATTERN = r"iteration_(\d+)"
BUNDLE_CONFIG_PATH = "memory_system_config.json"
MEMORY_CONFIG_PATH = "memory_config.json"
FEEDBACK_FILE = "feedbacks.json"
AGGREGATED_FEEDBACK_FILE = "aggregated_feedbacks.json"
UPDATED_VALUES_FILE = "update_results.json"


FEEDBACK_INSTRUCTION_TEMPLATE = """## Instruction 

Your task is to generate feedback for improving one target variable's value so the provided error case can be avoid.

Inputs you receive:
- Information about the target variable, including the variable's value, comment, category, etc.
- One failed cases including the attributed error type, attributed error reason, and an XML rendering of the operation graph around the attributed operation.
{history_input_bullet}

How to read the XML operation graph:
- The graph contains variables, edges, and operations.
- Variable comments describe what each variable represents, its intended role, and any local constraints on its value.
- Edge comments describe how information is transferred, transformed, filtered, or constrained between variables.
- Operation comments describe the operation's role, responsibility, and constraints in the workflow.
- Within one operation graph, variables with in-degree 0 relative to that graph are the operation's input variables.
- Variables with in-degree >= 1 relative to that graph are variables produced by the operation.
- Variables with out-degree 0 relative to that graph are the operation's final output variables.
- Use the operation name, category, comment, variable comments, edge comments, variable values, and produced variable roles to understand what the operation is supposed to do and how it fails.

Optimization boundary:
- The operation flow is not optimizable. 
- Only the current value of the target variable is optimizable.
- Therefore, every piece of feedback must be phrased as a concrete improvement suggestion to the target variable's value.

Feedback requirements:
- Every piece of feedback must be phrased as a concrete improvement suggestion to the target variable's value.
- Feedback should suggest how to improve the current value of the variable, rather than directly rewriting or replacing it with a completely new value.
- When providing feedback, pay close attention to the variable's comment and the context in which the variable is used within the workflow. Feedback must be realistic and actionable within the constraints and objectives of the current workflow or operation.
- Feedback should focus on improving the quality, clarity, correctness, or effectiveness of the variable value, instead of making unrelated or overly broad suggestions.

## Inputs 

### Target Variable

{target}

### Failed Case 

{failed_case}
{history_input_section}
"""


AGGREGATION_INSTRUCTION_TEMPLATE = """## Instruction 

Your task is to aggregate multiple feedback suggestions for improving one target variable.

Inputs you receive:
- A list of feedback suggestions collected from different failed cases.

Aggregation requirements:
- Merge overlapping or redundant feedback suggestions when possible.
- Preserve important details and constraints mentioned across different feedbacks. 
- The aggregated feedback should focus on improving the quality, clarity, correctness, or effectiveness of the variable value, instead of making overly broad suggestions. 

## Inputs

### Collected Feedback Suggestions 

{feedbacks}
"""


UPDATE_INSTRUCTION_TEMPLATE = """## Instruction 

Your task is to improve the current value of a target variable based on aggregated feedback suggestions.

Inputs you receive:
- Information about the target variable, including the variable's value, comment, category, etc.
- Aggregated feedback suggestions describing how the current value should be improved. 

Requirements:
- Improve the current target variable value according to the aggregated feedback suggestions.
- Parts of the current target variable value that are unrelated to the aggregated feedback suggestions should be preserved as much as possible.
- Ensure the improved value remains consistent with the variable's role, comment, and workflow context.

## Inputs 

### Target Variable Information 

{target_variable_info}

### Aggregated Feedback Suggestions

{aggregated_feedback}
"""


def find_last_n_iterations(
    folder: str | Path,
    n: int = 1,
    return_iteration_numbers: bool = False,
) -> list[str | tuple[int, str]] | None:
    """Get the last n valid iterations in the folder.

    Args:
        folder (`str`):
            The folder path to search for valid iterations.
        n (`int`, defaults to `1`):
            The number of iterations to get.
        return_iteration_numbers (`bool`, defaults to `False`):
            Whether to return the iteration numbers along with the paths.

    Returns:
        `list[str | tuple[int, str]] | None`:
            The paths to the last n valid iterations, or `None` if not found.
            If `return_iteration_numbers` is `True`, the iteration numbers will 
            be returned along with the paths.
    """
    if isinstance(folder, Path):
        folder = str(folder)
    if not os.path.isdir(folder):
        return None

    iteration_pattern = re.compile(ITERATION_PATTERN)
    iteration_dirs = []
    
    for path in os.listdir(folder):
        full_path = os.path.join(folder, path)
        target_config_path = os.path.join(full_path, BUNDLE_CONFIG_PATH)
        if os.path.isdir(full_path):
            match = iteration_pattern.match(path)
            if match and os.path.exists(target_config_path):
                step = int(match.group(1))
                iteration_dirs.append((step, full_path))
    
    if not iteration_dirs:
        return None
    
    # Sort by the iteration number and return the path with the highest iteration. 
    iteration_dirs.sort(key=lambda x: x[0])
    last_n_iterations = iteration_dirs[-n:]

    if return_iteration_numbers:
        return last_n_iterations
    return [path for _, path in last_n_iterations]


def create_iter_dir(iteration: int, parent_dir: str | Path | None = None) -> str:
    """Create an iteration directory and return its path.

    Args:
        iteration (`int`):
            An iteration number.
        parent_dir (`str | Path | None`, defaults to `None`):
            The parent directory of the iteration directory.

    Returns:
        `str`:
            The path to the created iteration directory.
    """
    if iteration < 1:
        raise ValueError(
            "The iteration number must be greater than 0. "
            f"However, the provided iteration number is '{iteration}'."
        )
    if parent_dir is not None:
        iter_dir = os.path.join(str(parent_dir), f"iteration_{iteration}") 
    else: 
        iter_dir = f"iteration_{iteration}"

    os.makedirs(iter_dir, exist_ok=True)
    return iter_dir


def load_dataset(
    dataset_path: str | Path, 
    dataset_type: str = "MemBase", 
    sample_size: int | None = None, 
    seed: int | None = None,
    save_path: str | Path | None = None,
) -> MemoryDataset: 
    """Load, optionally sample, and optionally save a dataset.

    Args:
        dataset_path (`str | Path`):
            The path to the dataset.
        dataset_type (`str`, defaults to `"MemBase"`):
            The dataset type key supported by MemBase.
        sample_size (`int | None`, optional):
            The number of trajectories to sample. If not provided, all
            trajectories are kept.
        seed (`int | None`, optional):
            The random seed used for sampling.
        save_path (`str | Path | None`, optional):
            If provided, it saves the loaded sampled dataset to this path. It 
            expects a file stem. 

    Returns:
        `MemoryDataset`:
            The loaded and sampled dataset.
    """
    if dataset_type not in DATASET_MAPPING:
        raise KeyError(
            f"`{dataset_type}` is not a valid dataset type. "
            f"Available dataset types are {list(DATASET_MAPPING.keys())}."
        )

    dataset_cls = DATASET_MAPPING[dataset_type]
    dataset_path = str(dataset_path)
    dataset = dataset_cls.read_raw_data(dataset_path)

    if sample_size is not None:
        dataset = dataset.sample(size=sample_size, seed=seed)

    if save_path is not None:
        save_path = Path(save_path)
        save_stem = save_path.with_suffix("")
        dataset.save_dataset(str(save_stem))
    return dataset
    

class MemoryConfigBundle(BaseModel):
    """A bundle of configurations for the memory system."""

    # It will be used to validate the assignment of the fields.
    model_config = ConfigDict(validate_assignment=True)

    _backend: Literal["Mem0"] = PrivateAttr(default="Mem0")

    # Preserve subclass-specific fields during serialization.
    base_config: SerializeAsAny[MemBaseConfig] = Field(
        ...,
        description="The base memory system configuration in MemBase.",
    ) 
    question_answering_prompt: str | None = Field(
        default=None,
        description="The question-answering template used in MemBase.",
    )

    @model_validator(mode="wrap")
    @classmethod
    def _restore_private_backend(
        cls,
        values: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Restore the private backend and deserialize the base config.

        Args:
            values (`Any`):
                The input values to validate.
            handler (`ModelWrapValidatorHandler[Self]`):
                The handler function to create the instance.

        Returns:
            `Self`:
                The validated instance with backend restored.
        """
        if not isinstance(values, dict):
            return handler(values)

        backend = values.get("backend", "Mem0")
        if backend not in CONFIG_MAPPING:
            raise ValueError(
                f"The provided memory system backend `{backend}` is not supported. "
                f"Available backends are {list(CONFIG_MAPPING.keys())}."
            )

        new_values = values.copy()
        new_values.pop("backend", None)
        config_cls = CONFIG_MAPPING[backend]
        if "base_config" in new_values and isinstance(new_values["base_config"], dict):
            new_values["base_config"] = config_cls.model_validate(new_values["base_config"])

        instance = handler(new_values)
        instance._backend = backend
        return instance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def backend(self) -> str:
        """Return the MemBase backend name.

        Returns:
            `str`:
                The backend name used to deserialize the base config.
        """
        return self._backend

    def save(self, folder: str | Path) -> None:
        """Save the bundle to the folder.

        Args:
            folder (`str | Path`):
                The folder to save the bundle.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / BUNDLE_CONFIG_PATH
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(mode="json"),
                f, 
                indent=4, 
                ensure_ascii=False,
            )

    def export_memory_config(self, folder: str | Path) -> None:
        """Export the memory config to the folder.

        Args:
            folder (`str | Path`):
                The folder where the memory config will be exported.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / MEMORY_CONFIG_PATH
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.base_config.model_dump(mode="json"),
                f,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        folder: str | Path,
    ) -> Self:
        """Load a memory config bundle from a folder.

        Args:
            folder (`str | Path`):
                The folder containing saved bundle files.

        Returns:
            `Self`:
                The loaded config bundle.
        """
        folder = Path(folder)
        config_path = folder / BUNDLE_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize a memory config bundle from a dictionary.

        Args:
            data (`dict[str, Any]`):
                The serialized memory config bundle.

        Returns:
            `Self`:
                The deserialized memory config bundle.
        """
        if "backend" not in data:
            raise ValueError("The backend is required.")
        return cls.model_validate(data)

    def get(self, key: str) -> Any:
        """Return a value by its key.

        Args:
            key (`str`):
                The key.

        Returns:
            `str | None`:
                The corresponding value.
        """
        if key == "question_answering_prompt":
            return self.question_answering_prompt
        return getattr(self.base_config, key)

    def update(self, key: str, value: Any) -> None:
        """Update one bundle field with Pydantic validation.

        Args:
            key (`str`):
                The field name to update.
            value (`Any`):
                The new field value.
        """   
        if key == "question_answering_prompt":
            self.question_answering_prompt = value
            return

        if key not in self.base_config.__class__.model_fields:
            raise KeyError(
                f"`{key}` is not a field of `{self.base_config.__class__.__name__}`."
            )
        data = self.base_config.model_dump(mode="python")
        data[key] = value
        self.base_config = self.base_config.__class__.model_validate(data)


class OptimizerConfig(AgentBaseConfig):
    """A configuration for the optimizer."""

    num_gradient_histories: int = Field(
        default=0,
        description="The number of text gradient histories to keep.",
        ge=0,
    )
    batch_size_for_summarization: int = Field(
        default=8,
        description="Number of feedbacks involved in each aggregation operation.",
        ge=1,
    )


class _FeedbackSuggestion(BaseModel):
    """Structured output for one target-variable feedback suggestion."""

    rationale: str = Field(
        description="Your thinking process for generating the feedback.",
    )
    feedback: str = Field(
        description="Concrete feedback for improving the target variable value.",
    )


class _AggregatedFeedbackSuggestion(BaseModel):
    """Structured output for aggregated target-variable feedback."""

    rationale: str = Field(
        description="Your thinking process for aggregating the feedback.",
    )
    aggregated_feedback: str = Field(
        description="Aggregated feedback for improving the target variable value.",
    )


class _TargetVariableUpdate(BaseModel):
    """Structured output for one updated target variable value."""

    rationale: str = Field(
        description="Your thinking process for updating the target variable value.",
    )
    improved_value: str = Field(
        description="The improved value that should replace the target variable value.",
    )


def _format_failed_case(
    error: ErrorAttributionPrediction,
    graph: ExecNetwork,
) -> str:
    """Format one attributed error as optimizer feedback input.

    Args:
        error (`ErrorAttributionPrediction`):
            The attributed error prediction.
        graph (`ExecNetwork`):
            The execution graph containing the attributed operation.

    Returns:
        `str`:
            Formatted failed-case text.
    """
    op_graph = graph.filter_by_operation(error.op_id)
    return (
        f"Error type: {error.error_type}\n"
        f"Attributed operation id: {error.op_id}\n"
        f"Reason: {error.reason}\n\n"
        "Erroneous operation's graph:\n"
        f"{op_graph.to_xml(include_metadata=False)}"
    )


def _format_history(history: deque[tuple[str, str]]) -> str:
    """Format optimization history records.

    Args:
        history (`deque[tuple[str, str]]`):
            Previous target values and feedback suggestions.

    Returns:
        `str`:
            Formatted optimization history.
    """
    if not history:
        return ""

    records = []
    for index, (value, feedback) in enumerate(history, start=1):
        records.append(
            f"#### History Record {index}\n"
            f"Previous Target Value:\n{value}\n\n"
            f"Feedback Suggestion:\n{feedback}"
        )
    return "\n\n".join(records)


def _format_feedbacks(feedbacks: list[str]) -> str:
    """Format feedback suggestions for aggregation.

    Args:
        feedbacks (`list[str]`):
            Feedback suggestions to aggregate.

    Returns:
        `str`:
            Formatted feedback list.
    """
    if not feedbacks: 
        raise ValueError(
            "There are no available feedbacks to aggregate."
        )

    records = []
    for index, feedback in enumerate(feedbacks, start=1): 
        records.append(
            f"#### Feedback {index}\n"
            f"Feedback content:\n{feedback}"
        )
    return "\n\n".join(records)


class OptimizerRunner(AgentBaseRunner):
    """A runner for the optimizer."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        """Initialize the optimizer runner.

        Args:
            config (`OptimizerConfig | None`, optional):
                The optimizer runner configuration. If not provided, 
                default configuration is used.
        """
        super().__init__(config or OptimizerConfig())
        self._gradient_histories = defaultdict(
            lambda: deque(maxlen=self.config.num_gradient_histories)
        ) 

    def _build_agent(
        self,
        name: str,
        sys_prompt: str,
        api_key: str,
        base_url: str,
    ) -> ReActAgent:
        """Build a plain structured-output optimizer agent.

        Args:
            name (`str`):
                Agent name.
            sys_prompt (`str`):
                System prompt for the agent.
            api_key (`str`):
                OpenAI-compatible API key.
            base_url (`str`):
                OpenAI-compatible base URL.

        Returns:
            `ReActAgent`:
                Configured optimizer agent.
        """
        cfg = self.config
        model = OpenAIChatModel(
            model_name=cfg.model,
            api_key=api_key,
            stream=cfg.stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": cfg.temperature},
        )
        return ReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=OpenAIChatFormatter(),
        )

    def _prepare_target_cases(
        self,
        errors: dict[str, list[ErrorAttributionPrediction]],
        graph: ExecNetwork | dict[str, list[ExecNetwork]],
    ) -> dict[str, list[tuple[ErrorAttributionPrediction, ExecNetwork]]]:
        """Pair each target error with its execution graph.

        Args:
            errors (`dict[str, list[ErrorAttributionPrediction]]`):
                Attributed errors grouped by target full node identifier.
            graph (`ExecNetwork | dict[str, list[ExecNetwork]]`):
                Shared graph or the execution graphs per target.

        Returns:
            `dict[str, list[tuple[ErrorAttributionPrediction, ExecNetwork]]]`:
                Target cases paired with their execution graphs.
        """
        if isinstance(graph, ExecNetwork):
            return {
                target: [(error, graph) for error in target_errors]
                for target, target_errors in errors.items()
            }

        target_cases = {}
        for target, target_errors in errors.items():
            if target not in graph:
                raise KeyError(
                    f"No execution graphs are provided for target `{target}`."
                )
            target_graphs = graph[target]
            if len(target_graphs) != len(target_errors):
                raise ValueError(
                    "Each target in `graph` must map to a list of execution "
                    "graphs with the same length as the corresponding errors. "
                    f"Target `{target}` has {len(target_errors)} errors and "
                    f"{len(target_graphs)} graphs."
                )
            target_cases[target] = list(zip(target_errors, target_graphs, strict=True))
        return target_cases

    async def arun(
        self,
        errors: dict[str, list[ErrorAttributionPrediction]],
        graph: ExecNetwork | dict[str, list[ExecNetwork]],
        batch_size: int | None = None,
        seed: int | None = None,
        save_folder: str | Path | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
        """Run the optimizer asynchronously.

        Args:
            errors (`dict[str, list[ErrorAttributionPrediction]]`):
                The errors used to generate the text gradients. 
                The key is the corresponding optimization target. It must 
                be a variable in the corresponding execution graph.
            graph (`ExecNetwork | dict[str, list[ExecNetwork]]`):
                The shared execution graph or the execution graphs per error.
            batch_size (`int | None`, optional):
                Maximum number of optimizer agents to run concurrently.
            seed (`int | None`, optional):
                The random seed used for shuffling the feedbacks.
            save_folder (`str | Path | None`, optional):
                The folder to save the optimization results. If not provided, 
                the optimization results will not be saved.

        Returns:
            `tuple[dict[str, str], dict[str, dict[str, float | int]]]`:
                Optimized target values and aggregate cost statistics.
        """
        if not errors:
            raise ValueError("No errors need to be optimized.")

        api_pool = self._api_pool
        batch_size = batch_size or 1
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if batch_size > api_pool.size:
            warnings.warn(
                "`batch_size` is greater than the available API credential slots.",
                UserWarning,
            )

        summarization_batch_size = self.config.batch_size_for_summarization

        target_cases = self._prepare_target_cases(errors, graph)
        for target, cases in target_cases.items():
            if not cases:
                raise ValueError(
                    f"No errors need to be optimized for target `{target}`."
                )

        semaphore = asyncio.Semaphore(batch_size)

        def summarize_phase_cost(
            elapsed_times: list[float],
            unit_count: int,
        ) -> dict[str, float | int]:
            """Summarize token and wall-clock cost for one optimizer phase.
            
            Args:
                elapsed_times (`list[float]`):
                    The elapsed seconds for each logical unit in the phase.
                unit_count (`int`):
                    The number of logical units in the phase.

            Returns:
                `dict[str, float | int]`:
                    The averaged token and time cost for the phase.
            """
            token_cost = ChatUsageTokenMonitor.to_dict()
            n_events = token_cost["n"]
            avg_input = token_cost["avg_input_tokens"]
            avg_output = token_cost["avg_output_tokens"]
            return {
                "average_input_tokens": n_events / max(1, unit_count) * avg_input,
                "average_output_tokens": n_events / max(1, unit_count) * avg_output,
                "average_minutes": sum(elapsed_times)
                / max(1, len(elapsed_times))
                / 60,
            }

        async def run_agent_call(
            call_index: int,
            name: str,
            sys_prompt: str,
            prompt: str,
            structured_model: type[BaseModel],
        ) -> tuple[BaseModel, float]:
            """Run one semaphore-guarded optimizer agent call.
            
            Args:
                call_index (`int`):
                    The index of the agent call in the batch.
                name (`str`):
                    The name of the agent.
                sys_prompt (`str`):
                    The system prompt for the agent.
                prompt (`str`):
                    The user prompt for the agent.
                structured_model (`type[BaseModel]`):
                    The structured model for the agent's response.

            Returns:
                `tuple[BaseModel, float]`:
                    The parsed structured response and elapsed seconds after 
                    acquiring the concurrency slot.
            """
            async with semaphore:
                started = time.perf_counter()
                api_key, base_url = api_pool.credential_for(call_index)
                agent = self._build_agent(
                    name=name,
                    sys_prompt=sys_prompt,
                    api_key=api_key,
                    base_url=base_url,
                )
                reply = await agent(
                    Msg("user", prompt, "user"),
                    structured_model=structured_model,
                )
                return (
                    structured_model.model_validate(reply.metadata),
                    time.perf_counter() - started,
                )

        async def generate_feedback(
            call_index: int,
            target: str,
            error: ErrorAttributionPrediction,
            case_graph: ExecNetwork,
        ) -> tuple[str, float]:
            """Generate feedback for one target-error pair
            
            Args:
                call_index (`int`):
                    The index of the agent call in the batch.
                target (`str`):
                    The target full node identifier.
                error (`ErrorAttributionPrediction`):
                    The attributed error prediction.
                case_graph (`ExecNetwork`):
                    The execution graph containing the attributed operation and 
                    the target variable.

            Returns:
                `tuple[str, float]`:
                    The generated feedback suggestion and elapsed seconds.
            """ 
            variable = case_graph.get_variable(target)
            target_info = variable.to_xml(
                include_metadata=False,
                include_variable_value=True,
            )
            history = _format_history(self._gradient_histories[target])
            history_input_bullet = ""
            history_input_section = ""

            # If the text gradient history is not empty, include it in the prompt.
            # This mimics the effect of momentum.
            if history:
                history_input_bullet = (
                    "\n- An optimization history of the target variable. Each "
                    "history record contains a previous value of the target "
                    "variable and a feedback suggestion that is provided for "
                    "improving that value."
                )
                history_input_section = (
                    "\n\n### Optimization History\n\n"
                    f"{history}"
                )
            prompt = FEEDBACK_INSTRUCTION_TEMPLATE.format(
                target=target_info,
                failed_case=_format_failed_case(error, case_graph),
                history_input_bullet=history_input_bullet,
                history_input_section=history_input_section,
            )
            response, elapsed = await run_agent_call(
                call_index=call_index,
                name="optimizer_feedback",
                sys_prompt=(
                    "You are an expert in generating concrete, valuable feedback " 
                    "for improving one optimizable target variable."
                ),
                prompt=prompt,
                structured_model=_FeedbackSuggestion,
            )
            return response.feedback, elapsed

        async def aggregate_feedbacks(
            call_index: int,
            feedbacks: list[str],
        ) -> tuple[str, float]:
            """Aggregate feedback suggestions for one target variable.
            
            Args:
                call_index (`int`):
                    The index of the agent call in the batch.
                feedbacks (`list[str]`):
                    The feedback suggestions to aggregate.

            Returns:
                `tuple[str, float]`:
                    The aggregated feedback suggestion and elapsed seconds.
            """
            if len(feedbacks) == 1:
                return feedbacks[0], 0.0

            shuffled = list(feedbacks)
            if seed is None:
                local_rng = random.Random()
            else:
                local_rng = random.Random(seed + call_index)
            local_rng.shuffle(shuffled)
            if summarization_batch_size == 1:
                summary = shuffled[0]
                elapsed = 0.0
                remaining_batches = [[feedback] for feedback in shuffled[1:]]
            else:
                summary_batch = shuffled[:summarization_batch_size]
                remaining = shuffled[summarization_batch_size:]
                # Create an initial summary from the first batch.
                summary, elapsed = await summarize_feedback_batch(
                    call_index=call_index,
                    feedbacks=summary_batch,
                )
                step = summarization_batch_size - 1
                remaining_batches = [
                    remaining[index:index + step]
                    for index in range(0, len(remaining), step)
                ]

            for batch in remaining_batches:
                summary, batch_elapsed = await summarize_feedback_batch(
                    call_index=call_index,
                    feedbacks=[summary, *batch],
                )
                elapsed += batch_elapsed
            return summary, elapsed

        async def summarize_feedback_batch(
            call_index: int,
            feedbacks: list[str],
        ) -> tuple[str, float]:
            """Summarize one feedback batch.
            
            Args:
                call_index (`int`):
                    The index of the agent call in the batch.
                feedbacks (`list[str]`):
                    The feedback suggestions to summarize.

            Returns:
                `tuple[str, float]`:
                    The summarized feedback suggestion and elapsed seconds.
            """
            prompt = AGGREGATION_INSTRUCTION_TEMPLATE.format(
                feedbacks=_format_feedbacks(feedbacks),
            )
            response, elapsed = await run_agent_call(
                call_index=call_index,
                name="optimizer_feedback_aggregator",
                sys_prompt=(
                    "You are an expert in aggregating feedback suggestions " 
                    "into a concise, non-redundant feedback for one target variable."
                ),
                prompt=prompt,
                structured_model=_AggregatedFeedbackSuggestion,
            )
            return response.aggregated_feedback, elapsed

        async def update_target(
            call_index: int,
            target: str,
            target_graph: ExecNetwork,
            aggregated_feedback: str,
        ) -> tuple[str, float]:
            """Apply aggregated feedback to one target variable value.
            
            Args:
                call_index (`int`):
                    The index of the agent call in the batch.
                target (`str`):
                    The target full node identifier.
                target_graph (`ExecNetwork`):
                    The execution graph containing the target variable.
                aggregated_feedback (`str`):
                    The aggregated feedback suggestion.

            Returns:
                `tuple[str, float]`:
                    The updated target variable value and elapsed seconds.
            """
            variable = target_graph.get_variable(target)
            target_info = variable.to_xml(
                include_metadata=False,
                include_variable_value=True,
            )
            prompt = UPDATE_INSTRUCTION_TEMPLATE.format(
                target_variable_info=target_info,
                aggregated_feedback=aggregated_feedback,
            )
            response, elapsed = await run_agent_call(
                call_index=call_index,
                name="optimizer_target_updater",
                sys_prompt=(
                    "You are an expert in improving one target variable value " 
                    "while preserving its role based on the provided feedback."
                ),
                prompt=prompt,
                structured_model=_TargetVariableUpdate,
            )
            return response.improved_value, elapsed

        studio = None
        if batch_size == 1 and self.config.studio_url is not None:
            studio = StudioServer(
                url=self.config.studio_url,
                project=self.config.project,
            )
            studio.activate()

        costs = {
            "feedback_generation": {
                "average_input_tokens": 0.0,
                "average_output_tokens": 0.0,
                "average_minutes": 0.0,
            },
            "feedback_aggregation": {
                "average_input_tokens": 0.0,
                "average_output_tokens": 0.0,
                "average_minutes": 0.0,
            },
            "update": {
                "average_input_tokens": 0.0,
                "average_output_tokens": 0.0,
                "average_minutes": 0.0,
            },
        }
        if save_folder is not None:
            save_folder = str(save_folder)
            os.makedirs(save_folder, exist_ok=True)

        try:
            with agentscope_token_monitor():
                feedback_tasks = []
                call_index = 0
                for target, cases in target_cases.items():
                    for error, case_graph in cases:
                        feedback_tasks.append(
                            (
                                target,
                                generate_feedback(
                                    call_index=call_index,
                                    target=target,
                                    error=error,
                                    case_graph=case_graph,
                                ),
                            )
                        )
                        call_index += 1

                ChatUsageTokenMonitor.reset()
                feedback_results_with_times = await asyncio.gather(
                    *[task for _, task in feedback_tasks]
                )
                feedback_results = [
                    feedback for feedback, _ in feedback_results_with_times
                ]
                feedback_times = [
                    elapsed for _, elapsed in feedback_results_with_times
                ]
                costs["feedback_generation"] = summarize_phase_cost(
                    elapsed_times=feedback_times,
                    unit_count=len(feedback_tasks),
                )

                target_feedbacks = defaultdict(list)
                for (target, _), feedback in zip(
                    feedback_tasks,
                    feedback_results,
                    strict=True,
                ):
                    target_feedbacks[target].append(feedback)

                if save_folder is not None:
                    with open(
                        os.path.join(save_folder, FEEDBACK_FILE), 
                        "w", 
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            {
                                target: feedbacks
                                for target, feedbacks in target_feedbacks.items()
                            },
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )

                ChatUsageTokenMonitor.reset()
                aggregated_pairs_with_times = await asyncio.gather(
                    *[
                        aggregate_feedbacks(
                            call_index=call_index + index,
                            feedbacks=feedbacks,
                        )
                        for index, (_, feedbacks)
                        in enumerate(target_feedbacks.items())
                    ]
                )
                aggregated_pairs = [
                    feedback for feedback, _ in aggregated_pairs_with_times
                ]
                aggregation_times = [
                    elapsed for _, elapsed in aggregated_pairs_with_times
                ]
                costs["feedback_aggregation"] = summarize_phase_cost(
                    elapsed_times=aggregation_times,
                    unit_count=len(target_feedbacks),
                )
                aggregated_feedbacks = {
                    target: feedback
                    for target, feedback in zip(
                        target_feedbacks.keys(),
                        aggregated_pairs,
                        strict=True,
                    )
                }
                call_index += len(target_feedbacks)

                if save_folder is not None:
                    with open(
                        os.path.join(
                            save_folder, 
                            AGGREGATED_FEEDBACK_FILE,
                        ), 
                        "w", 
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            aggregated_feedbacks,
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )

                ChatUsageTokenMonitor.reset()
                updated_values_with_times = await asyncio.gather(
                    *[
                        update_target(
                            call_index=call_index + index,
                            target=target,
                            target_graph=target_cases[target][0][1],
                            aggregated_feedback=aggregated_feedback,
                        )
                        for index, (target, aggregated_feedback)
                        in enumerate(aggregated_feedbacks.items())
                    ]
                )
                updated_values = [
                    value for value, _ in updated_values_with_times
                ]
                update_times = [
                    elapsed for _, elapsed in updated_values_with_times
                ]
                costs["update"] = summarize_phase_cost(
                    elapsed_times=update_times,
                    unit_count=len(aggregated_feedbacks),
                )
        finally:
            if studio is not None:
                studio.deactivate()

        updates = {
            target: value
            for target, value in zip(
                aggregated_feedbacks.keys(),
                updated_values,
                strict=True,
            )
        }
        for target, aggregated_feedback in aggregated_feedbacks.items():
            variable = target_cases[target][0][1].get_variable(target)
            target_value = variable.value
            self._gradient_histories[target].append(
                (target_value, aggregated_feedback)
            )
        ChatUsageTokenMonitor.reset()

        if save_folder is not None:
            with open(
                os.path.join(
                    save_folder, 
                    UPDATED_VALUES_FILE,
                ), 
                "w", 
                encoding="utf-8",
            ) as f:
                json.dump(
                    updates,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

        return updates, costs

    def run(
        self,
        errors: dict[str, list[ErrorAttributionPrediction]],
        graph: ExecNetwork | dict[str, list[ExecNetwork]],
        batch_size: int | None = None,
        seed: int | None = None,
        save_folder: str | Path | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
        """Run the optimizer.

        Args:
            errors (`dict[str, list[ErrorAttributionPrediction]]`):
                The errors used to generate the text gradients.
            graph (`ExecNetwork | dict[str, list[ExecNetwork]]`):
                The shared execution graph or the execution graphs per error.
            batch_size (`int | None`, optional):
                Maximum number of optimizer agents to run concurrently.
            seed (`int | None`, optional):
                The random seed used for shuffling the feedbacks.
            save_folder (`str | Path | None`, optional):
                The folder to save the optimization results. If not provided, 
                the optimization results will not be saved.

        Returns:
            `tuple[dict[str, str], dict[str, dict[str, float | int]]]`:
                Optimized target values and aggregate cost statistics.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    errors=errors,
                    graph=graph,
                    batch_size=batch_size,
                    seed=seed,
                    save_folder=save_folder,
                )
            )

        raise RuntimeError(
            "`OptimizerRunner(...).run(...)` cannot be called from a running "
            "event loop. Use `await OptimizerRunner(...).arun(...)` instead."
        )