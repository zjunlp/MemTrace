import json
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field,
)
from tqdm import tqdm
from smartcomment import (
    comment_graph,
    comment_session,
    enable_tracing,
    disable_tracing,
)
from ..configs import CONFIG_MAPPING
from ..datasets import DATASET_MAPPING, ONLINE_EVAL_ENV_MAPPING
from ..datasets.online_base import OnlineMemBaseDataset, OnlineEvalEnv
from ..layers import MEMORY_LAYERS_MAPPING
from ..model_types.dataset import (
    Message, 
    Trajectory, 
    QuestionAnswerPair,
    MemoryDataset, 
)
from ..model_types.evaluation import OnlineEvalResult
from ..utils import (
    CostState,
    CostStateManager,
    MonkeyPatcher,
    get_tokenizer_for_model, 
)
from typing import Any, Callable


_LOCK = threading.Lock()


def memory_construction(
    layer_type: str, 
    user_id: str, 
    trajectory: Trajectory, 
    config: dict[str, Any] | None = None, 
    rerun: bool = False,
    message_preprocessor: Callable[[Message], Message] | None = None,
    strict: bool = True,
    traced_data_save_dir: str | None = None,
    dataset_cls: type[MemoryDataset] | None = None,
    online_eval_env: OnlineEvalEnv | None = None,
) -> dict[str, float | list[OnlineEvalResult]]: 
    """Given a specific interaction trajectory, build a memory for one user.

    This function is designed to be submitted to a thread pool so that
    multiple trajectories can be processed in parallel.

    Args:
        layer_type (`str`): 
            The registered name of the memory layer.
        user_id (`str`): 
            Unique identifier of the user whose memory is being built.
        trajectory (`Trajectory`): 
            The interaction trajectory consisting of sessions and messages.
        config (`dict[str, Any] | None`, optional): 
            Raw configuration dictionary for the memory layer. If not provided, 
            defaults will be used.
        rerun (`bool`, defaults to `False`): 
            If `False`, skip construction when a saved memory already exists.
        message_preprocessor (`Callable[[Message], Message]`, optional): 
            A callable that preprocesses each message before it is added to the memory.
        strict (`bool`, defaults to `True`):
            If it is enabled, any error raised by the memory layer during memory construction 
            will propagate and abort the trajectory. If it is disabled, such errors are logged 
            and the message is skipped so the remaining trajectory can continue.
        traced_data_save_dir (`str | None`, optional):
            Directory where execution graph artefacts are saved. If not provided,
            the graph saving is skipped.
        dataset_cls (`type[MemoryDataset] | None`, optional):
            The dataset class. When it is an online dataset, the memory construction process is 
            routed to its evaluation logic for task messages.
        online_eval_env (`OnlineEvalEnv | None`, optional):
            Evaluation environment for online datasets. If not provided, online
            evaluation is skipped even for task messages.

    Returns:
        `dict[str, float | list[OnlineEvalResult]]`: 
            A dictionary containing the total time and the average time per operation of 
            adding new message. If the dataset is an online dataset, the evaluation results are 
            also included.
    """
    config = deepcopy(config) or {}
    # It overrides the user id in the config. 
    config["user_id"] = user_id 
    # Each user has a distinct config directory.
    config["save_dir"] = f"{config['save_dir']}/{user_id}" 

    # Use lazy mapping to load config and layer classes.
    config_cls = CONFIG_MAPPING[layer_type]
    config = config_cls(**config)
    layer_cls = MEMORY_LAYERS_MAPPING[layer_type] 
    
    with _LOCK:
        layer = layer_cls(config)

    if message_preprocessor is None:
        message_preprocessor = lambda message: message

    is_online = (
        dataset_cls is not None
        and issubclass(dataset_cls, OnlineMemBaseDataset)
        and online_eval_env is not None
    )

    output = {
        "total_add_time": 0.0,
        "avg_add_time": 0.0,
        "eval_results": [],
    }

    with _LOCK:
        # It includes I/O operations. 
        if not rerun and layer.load_memory(user_id):
            print(f"🔄 The memory for user '{user_id}' is loaded successfully 😄.")
            return output
    
    specs = layer.get_patch_specs() 

    with comment_graph(
        user_id=user_id, 
        project_id=layer_type, 
        strict=True,
        metadata={
            "save_dir": config.save_dir, 
            "layer_type": layer_type, 
            "traj_metadata": trajectory.metadata,
            "traced_data_save_dir": traced_data_save_dir,
            "dataset_cls": dataset_cls.__name__ if dataset_cls is not None else None,
        },
    ) as graph:
        with MonkeyPatcher(specs):
            total_msgs = sum(len(session) for session in trajectory) 
            pbar_desc = f"🧑 User Identifier: {user_id} | 🧠 Memory Layer Type: {layer_type}" 
            pbar = tqdm(
                total=total_msgs,
                desc=pbar_desc,
                leave=False,   
            )

            num_add_failed = 0
            num_eval_failed = 0
            for i, session in enumerate(trajectory, start=1):
                with comment_session(
                    session_id=session.id, 
                    comment=f"It is the {i}th conversational session.", 
                    category="memory_construction",
                    metadata=session.metadata,
                ):
                    for message in session:
                        start_time = datetime.now() 
                        message = message.model_copy(deep=True)
                        message = message_preprocessor(message)

                        try:
                            if message.metadata.get("is_task"):
                                if is_online:
                                    eval_result = dataset_cls.online_evaluate(
                                        message, 
                                        layer, 
                                        online_eval_env,
                                    )
                                    output["eval_results"].extend(eval_result)
                                else:
                                    if dataset_cls is not None:
                                        warnings.warn(
                                            f"Message '{message.id}' is marked as a task but "
                                            f"dataset '{dataset_cls.__name__}' does not support "
                                            "online evaluation. It falls back to normal memory construction.",
                                            UserWarning,
                                        )
                                    layer.add_message(message)
                            else:
                                layer.add_message(message)
                        except Exception as e:
                            if strict:
                                pbar.close()
                                raise
                            if message.metadata.get("is_task") and is_online:
                                num_eval_failed += 1
                                print(
                                    "⚠️ Online evaluation fails for a task message "
                                    f"'{message.id}' for user '{user_id}': "
                                    f"{e.__class__.__name__}: {e}"
                                )
                            else:
                                num_add_failed += 1
                                print(
                                    "⚠️ The message is skipped because the memory layer "
                                    f"fails to add it for user '{user_id}': "
                                    f"{e.__class__.__name__}: {e}"
                                )

                        end_time = datetime.now() 
                        output["total_add_time"] += (end_time - start_time).total_seconds()

                        pbar.update(1) 
                        time.sleep(0.2)
            pbar.close()

            total_failed = num_add_failed + num_eval_failed
            if total_failed > 0:
                parts = []
                if num_add_failed > 0:
                    parts.append(f"{num_add_failed} message addition failure(s)")
                if num_eval_failed > 0:
                    parts.append(f"{num_eval_failed} online evaluation failure(s)")
                print(
                    f"⚠️ {total_failed}/{total_msgs} messages are not processed successfully "
                    f"for user '{user_id}': {', '.join(parts)}."
                )
     
        with comment_session(
            comment=(
                "It is the finalization stage of memory construction. " 
                "This stage ensures all messages have been fully ingested " 
                "into the memory layer and the internal memory state is consistent."
            ),
            category="memory_construction",
        ):
            start_time = datetime.now()   
            # It may include I/O operations (loading a sentence embedding model).
            with _LOCK:
                layer.flush()
            end_time = datetime.now()  
            output["total_add_time"] += (end_time - start_time).total_seconds() 

    # It includes I/O operations. 
    with _LOCK:
        layer.save_memory() 
        layer.cleanup()
    output["avg_add_time"] = output["total_add_time"] / len(trajectory)

    # Finally, if the tracing is enabled, we export the graph to a JSON file.
    if graph is not None and traced_data_save_dir is not None:
        graph_data = graph.export_graph()
        traced_data_path = os.path.join(
            traced_data_save_dir,
            user_id,
            "graph_construction.json",
        )
        os.makedirs(
            os.path.dirname(traced_data_path), 
            exist_ok=True
        )
        with open(
            traced_data_path, 
            "w", 
            encoding="utf-8",
        ) as f:
            json.dump(
                graph_data, 
                f, 
                indent=4, 
                ensure_ascii=False,
            )

    return output 


class ConstructionRunnerConfig(BaseModel):
    """Configuration for the construction runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_type: str = Field(
        ..., 
        description="The type of the memory layer to be evaluated.",
    )
    dataset_type: str = Field(
        ..., 
        description="The type of the dataset used to evaluate the memory layer.",
    )
    dataset_path: str = Field(
        ..., 
        description="The path to the dataset.",
    )
    dataset_standardized: bool = Field(
        default=False,
        description="Whether the dataset is already standardized.",
    )
    config_path: str | None = Field(
        default=None,
        description="Path to the JSON config for the memory method.",
    )
    memory_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dictionary config for the memory method. "
            "If provided, it takes precedence over ``config_path``."
        ),
    )
    num_workers: int = Field(
        default=4,
        description="The number of threads to use for the evaluation.",
    )
    seed: int = Field(
        default=42,
        description=(
            "Random seed used to sample the dataset if the user provides the sample size."
        ),
    )
    sample_size: int | None = Field(
        default=None,
        description="Subset size from dataset.",
    )
    rerun: bool = Field(
        default=False,
        description="Ignore saved memory and rebuild the memory from scratch.",
    )
    strict: bool = Field(
        default=True,
        description=(
            "If it is enabled, any error raised by the memory layer during memory construction "
            "will propagate and abort the trajectory. If it is disabled, such errors are logged "
            "and the message is skipped so the remaining trajectory can continue."
        ),
    )
    token_cost_save_filename: str = Field(
        default="token_cost",
        description="Path to save the statistics related to the token consumption.",
    )
    start_idx: int | None = Field(
        default=None,
        description="The starting index of the trajectories to be processed.",
    )
    end_idx: int | None = Field(
        default=None,
        description="The ending index of the trajectories to be processed.",
    )
    tokenizer_path: str | None = Field(
        default=None,
        description="The path to the tokenizer (only for backbone model).",
    )
    message_preprocessor: Callable[[Message], Message] | None = Field(
        default=None,
        description=(
            "A callable that preprocesses each message before it is added to the memory."
        ),
    )
    sample_filter: Callable[[Trajectory, list[QuestionAnswerPair]], bool] | None = Field(
        default=None,
        description="A callable that filters dataset samples.",
    )
    online_eval_config_path: str | None = Field(
        default=None,
        description="Path to a JSON config for the online evaluation environment.",
    )
    traced_data_save_dir: str = Field(
        default="traced_data",
        description="Directory where execution graph artefacts are saved.",
    )
    tracing: bool = Field(
        default=False,
        description=(
            "Whether to enable execution graph tracing. "
            "Note that this only applies to memory systems that currently support tracing."
        ),
    )


class ConstructionRunner:
    """The runner that orchestrates the memory construction stage.

    It loads the dataset, sets up token-cost monitoring, and dispatches
    per-trajectory construction tasks to a thread pool. When tracing is
    enabled, it records the memory construction flow for memory systems that
    currently support traced execution. For online-capable datasets, it also
    resolves the evaluation environment and collects, aggregates, and persists
    evaluation results.
    """

    def __init__(self, config: ConstructionRunnerConfig) -> None:
        """Initialize the construction runner.

        Args:
            config (`ConstructionRunnerConfig`): 
                The runner configuration.
        """
        self.config = config

    def _resolve_memory_config(self) -> dict[str, Any] | None:
        """Return the memory configuration dictionary."""
        if self.config.memory_config is not None:
            return self.config.memory_config
        if self.config.config_path is not None:
            with open(self.config.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def run(self) -> list[dict[str, float | list[OnlineEvalResult]]]:
        """Execute the memory construction pipeline.

        Returns:
            `list[dict[str, float | list[OnlineEvalResult]]]`: 
                A list of per-trajectory result dictionaries containing the total time and 
                the average time per operation of adding new message. 
                If the dataset is an online dataset, the evaluation results are also included.
        """
        cfg = self.config
        config = self._resolve_memory_config()

        # Get a dummy configuration to infer the corresponding LLM being used.
        # Use lazy mapping to load config class.
        config_cls = CONFIG_MAPPING[cfg.memory_type]
        if config is None:
            dummy_config = config_cls(user_id="guest")
        else:
            dummy_config = config_cls(**config)

        # If token cost file exists, load it.
        if os.path.exists(cfg.token_cost_save_filename + ".json"):
            with open(cfg.token_cost_save_filename + ".json", "r") as f:
                token_cost = json.load(f)
            for model, state in token_cost.items():
                is_dict = all(isinstance(value, dict) for value in state.values())
                if not is_dict:
                    token_cost[model] = CostState.from_dict(state)
                else:
                    token_cost[model] = {
                        op: CostState.from_dict(cs) for op, cs in state.items()
                    }
        else:
            token_cost = {}

        # Before run the expriment, we should register the base model being used. 
        # Please ensure all types of config classes have a `get_llm_models` method. 
        # The tokenizer is inferred from the model name automatically. 
        llm_models = dummy_config.get_llm_models()
        if cfg.tokenizer_path is not None:
            tokenizer = get_tokenizer_for_model(cfg.tokenizer_path)
        else:
            tokenizer = None

        for llm_model in llm_models:
            state = token_cost.get(llm_model)
            if state is not None:
                print(
                    f"There is a saved checkpoint 📁 for monitoring the token consumption of {llm_model} 🤖. "
                    "It will be loaded into `CostStateManager`."
                )
            CostStateManager.register(llm_model, state=state, tokenizer=tokenizer)
        del dummy_config
        if len(llm_models) > 0:
            print(
                f"The LLM model(s) 🤖 being used are {', '.join(llm_models)}. "
                "They have been registered in `CostStateManager`."
            )
            print()

        # Prepare the dataset using lazy mapping.
        ds_cls = DATASET_MAPPING[cfg.dataset_type]
        if cfg.dataset_standardized:
            dataset = ds_cls.read_dataset(cfg.dataset_path)
        else:
            dataset = ds_cls.read_raw_data(cfg.dataset_path)
        if cfg.sample_size is not None or cfg.sample_filter is not None:
            dataset = dataset.sample(
                size=cfg.sample_size,
                seed=cfg.seed,
                sample_filter=cfg.sample_filter,
            )
            dataset.save_dataset(f"{config['save_dir']}/{cfg.dataset_type}_stage_1")
        print("The dataset is loaded successfully 😄.")
        # `print(dataset)` calls the __str__ method defined by Pydantic's BaseModel.
        print(repr(dataset))
        print()

        # Resolve online evaluation environment.
        online_eval_env = None
        if (
            issubclass(ds_cls, OnlineMemBaseDataset)
            and cfg.dataset_type in ONLINE_EVAL_ENV_MAPPING
        ):
            env_cls = ONLINE_EVAL_ENV_MAPPING[cfg.dataset_type]
            if cfg.online_eval_config_path is not None:
                with open(cfg.online_eval_config_path, "r", encoding="utf-8") as f:
                    env_config = json.load(f)
                online_eval_env = env_cls(**env_config)
            else:
                online_eval_env = env_cls()
            print(
                f"⚖️ Online evaluation is enabled for the dataset '{cfg.dataset_type}'."
            )
            print()

        # Resolve index range.
        start_idx = cfg.start_idx if cfg.start_idx is not None else 0
        end_idx = cfg.end_idx if cfg.end_idx is not None else len(dataset)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(dataset))
        if start_idx >= end_idx:
            raise ValueError("The starting index must be less than the ending index.")

        if cfg.tracing:
            enable_tracing()
        else:
            disable_tracing()

        # Dispatch per-trajectory construction to a thread pool.
        results = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            future_to_user_id = {}
            for trajectory, _ in zip(*dataset[start_idx:end_idx]):
                # Note that this code is for academic purpose, the embedding model may be loaded multiple times. 
                user_id = trajectory.id
                future = executor.submit(
                    memory_construction,
                    cfg.memory_type,
                    user_id,
                    trajectory,
                    config=config,
                    rerun=cfg.rerun,
                    message_preprocessor=cfg.message_preprocessor,
                    strict=cfg.strict,
                    traced_data_save_dir=cfg.traced_data_save_dir,
                    dataset_cls=ds_cls,
                    online_eval_env=online_eval_env,
                )
                future_to_user_id[future] = user_id
            for future in tqdm(
                as_completed(future_to_user_id),
                total=len(future_to_user_id),
                desc="📉 Processing trajectories",
            ):
                try:
                    result = future.result()
                    result["user_id"] = future_to_user_id[future]
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error occurs when the trajectory is processed: {e}")

        if len(results) == end_idx - start_idx:
            print("The memory construction process is completed successfully 😀.")

        # Print statistics.
        total_time = 0.0
        avg_time_per_add_session = 0.0
        num_valid_trajectories = 0
        for result in results:
            # Statistics on the newly processed trajectories. 
            if result["total_add_time"] > 0:
                total_time += result["total_add_time"]
                avg_time_per_add_session += result["avg_add_time"]
                num_valid_trajectories += 1
        avg_time = total_time / max(num_valid_trajectories, 1)
        avg_time_per_add_session = avg_time_per_add_session / max(num_valid_trajectories, 1)
        print(
            f"For {cfg.memory_type}, the average time per trajectory "
            f"({num_valid_trajectories} in {len(results)}) is {avg_time:.2f} seconds."
        )
        print(
            f"For {cfg.memory_type}, the average time per operation of adding new session "
            f"is {avg_time_per_add_session:.2f} seconds."
        )

        # Aggregate and save online evaluation results.
        if online_eval_env is not None:
            all_eval_entries = []
            metric_sums = {}
            metric_counts = {}
            for result in results:
                result_user_id = result.get("user_id")
                for eval_result in result.get("eval_results", []):
                    metrics = eval_result["metrics"]
                    rollout = eval_result["rollout"]
                    for metric_name, metric_val in metrics.items():
                        # The available metrics are subject to the specific question type and may vary.
                        metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + metric_val["value"]
                        metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

                    all_eval_entries.append(
                        {
                            "user_id": result_user_id,
                            "metrics": metrics,
                            "rollout": [
                                msg.model_dump(mode="python") for msg in rollout
                            ],
                        }
                    )

            # If all messages have been skipped or all evaluation processes fail
            # no evaluation results are reported.
            if not all_eval_entries:
                print("No online evaluation results are reported.")
            else:
                # Print online evaluation summary.
                print()
                print(f"📊 Online evaluation summary:")
                print(f"There are {len(all_eval_entries)} task(s) are evaluated.")
                for metric_name in sorted(metric_sums):
                    avg = metric_sums[metric_name] / metric_counts[metric_name]
                    print(f"  {metric_name}: {avg:.4f} ({metric_counts[metric_name]} samples)")
                print()

                # Save online evaluation results.
                save_dir = config['save_dir']
                output_path = os.path.join(
                    save_dir,
                    f"{cfg.dataset_type}_online_evaluation.json",
                )
                os.makedirs(save_dir, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        all_eval_entries, 
                        f, 
                        ensure_ascii=False, 
                        indent=4,
                    )
                print(
                    f"✅ {len(all_eval_entries)} online evaluation results are saved to '{output_path}'." 
                )

        # Save token cost statistics.
        CostStateManager.save_to_json_file(cfg.token_cost_save_filename)
        CostStateManager.reset()

        return results
