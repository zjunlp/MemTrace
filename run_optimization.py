# -*- coding: utf-8 -*-
"""Automatically optimize Mem0 prompts on LoCoMo.

It runs Mem0 on a sampled LoCoMo split, finds failed judged queries from the 
exported smartcomment graph, attributes each failure with MemTrace, and updates 
the three configurable prompts.
"""

import argparse
import asyncio
import json
import os
import warnings
from string import Template
from membase.runners.construction import (
    ConstructionRunner,
    ConstructionRunnerConfig,
)
from membase.runners.evaluation import (
    EvaluationRunner,
    EvaluationRunnerConfig,
)
from membase.runners.search import (
    SearchRunner,
    SearchRunnerConfig,
)
from smartcomment.runtime import ExecNetwork 
from toolkits.optimization_utils import (
    MemoryConfigBundle, 
    find_last_n_iterations,
    create_iter_dir, 
    load_dataset, 
)
from toolkits.bench_utils import FailedQueryCase, collect_failed_queries
from toolkits.memtrace_utils import (
    ErrorAttributionPrediction, 
    MemTraceRunner, 
    MemTraceConfig,
)
from toolkits.optimization_utils import OptimizerConfig, OptimizerRunner 


MEMORY_TYPE = "Mem0"
DATASET_TYPE = "LoCoMo"
SAMPLED_DATASET_FILE = "sampled_dataset.json"
MEMROY_TOKEN_COST_FILENAME = "memory_construction_token_cost"
FAILED_CASES_FILE = "failed_cases.json"
ERROR_ATTRIBUTION_FILE = "error_attribution.json"
SUMMARY_FILE = "iteration_summary.json"

# A mappping from the configuration bundle field name 
# to the corresponding variable name in the execution graph.
PROMPT_TARGETS = {
    "custom_fact_extraction_prompt": "fact-extraction-system-prompt",
    "custom_update_memory_prompt": "memory-update-decision-prompt",
    "question_answering_prompt": "question-answering-prompt",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        `argparse.Namespace`:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automatically optimize Mem0 prompts on LoCoMo.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the initial Mem0 JSON config.",
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the raw LoCoMo dataset.",
        required=True,
    )
    parser.add_argument(
        "--optimization-dir",
        type=str,
        default="auto_optimization",
        help=(
            "Directory where all sampled datasets, iteration folders, execution "
            "graphs, prompt bundles, attribution results, and cost summaries are saved."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample and order the LoCoMo trajectories.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of LoCoMo trajectories to sample before optimization.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve for each query during memory search.",
    )
    parser.add_argument(
        "--api-config-path",
        type=str,
        default=None,
        help=(
            "Path to an API config JSON with `api_keys` and `base_urls`. "
            "If not provided, the optimizer falls back to OpenAI-compatible "
            "environment variables for attribution and prompt optimization. "
            "Evaluation receives this file only when it is explicitly provided."
        ),
    )
    parser.add_argument(
        "--qa-model",
        default="gpt-4.1-mini",
        help="Model name for question answering.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Model name for LLM-as-a-judge.",
    )
    parser.add_argument(
        "--attribution-model",
        default="gpt-4.1",
        help="Model name used by MemTrace for error attribution.",
    )
    parser.add_argument(
        "--disable-memtrace",
        action="store_true",
        help=(
            "Disable MemTrace and uniformly attribute every failed case as a "
            "reponse error to all optimizable prompt targets."
        ),
    )
    parser.add_argument(
        "--optimizer-model",
        default="gpt-4.1",
        help=(
            "Model name used to aggregate text gradients " 
            "and rewrite the optimizable prompts."
        ),
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=4,
        help="Batch size for question-answering evaluation.",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=4,
        help="Batch size for LLM-as-a-judge evaluation.",
    )
    parser.add_argument(
        "--attribution-batch-size",
        type=int,
        default=4,
        help="Maximum number of failed-query attribution agents to run concurrently.",
    )
    parser.add_argument(
        "--optimization-batch-size",
        type=int,
        default=4,
        help="Maximum number of optimizer agents to run concurrently.",
    )
    parser.add_argument(
        "--num-gradient-histories",
        type=int,
        default=1,
        help="Number of previous text-gradient histories to include during optimization.",
    )
    parser.add_argument(
        "--batch-size-for-summarization",
        type=int,
        default=4,
        help="Number of feedback suggestions included in each aggregation prompt.",
    )
    parser.add_argument(
        "--max-trace-nodes",
        type=int,
        default=16,
        help="Maximum number of graph trace nodes the MemTrace notebook may explore.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum reasoning iterations per MemTrace attribution case.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=272_000,
        help="Context-window threshold used by AgentScope formatter compression.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Ignore existing checkpoints and rerun from the first sampled trajectory. "
            "By default, the optimizer automatically resumes from the latest iteration."
        ),
    )
    return parser.parse_args()


def generate_execution_graph(
    args: argparse.Namespace,
    iter_dir: str,
    sampled_index: int,
    bundle: MemoryConfigBundle,
) -> dict[str, str | dict[str, float] | int]:
    """Generate an execution graph for one sampled trajectory.

    Args:
        args (`argparse.Namespace`):
            Parsed CLI arguments.
        iter_dir (`str`):
            Iteration directory.
        sampled_index (`int`):
            Sampled trajectory index.
        bundle (`MemoryConfigBundle`):
            Memory config bundle used by this iteration.

    Returns:
        `dict[str, str | dict[str, float] | int]`:
            A dictionary containing the path to the final execution graph, 
            the metrics, and the number of queries.
    """
    traced_dir = os.path.join(iter_dir, "traced_data")
    token_cost_save_filename = os.path.join(iter_dir, MEMROY_TOKEN_COST_FILENAME)
    current_memory_config = bundle.base_config.model_dump(mode="python")

    construction_config = ConstructionRunnerConfig(
        memory_type=MEMORY_TYPE,
        dataset_type=DATASET_TYPE,
        dataset_path=args.dataset_path,
        dataset_standardized=True,
        memory_config=current_memory_config,
        num_workers=1,
        rerun=True, 
        strict=True,
        token_cost_save_filename=token_cost_save_filename,
        start_idx=sampled_index,
        end_idx=sampled_index + 1,
        traced_data_save_dir=traced_dir,
        tracing=True,
    )
    construction_results = ConstructionRunner(construction_config).run()
    user_id = construction_results[0]["user_id"]

    search_config = SearchRunnerConfig(
        memory_type=MEMORY_TYPE,
        dataset_type=DATASET_TYPE,
        dataset_path=args.dataset_path,
        dataset_standardized=True, 
        top_k=args.top_k,
        strict=True, 
        memory_config=current_memory_config,
        num_workers=1,
        start_idx=sampled_index,
        end_idx=sampled_index + 1,
        traced_data_save_dir=traced_dir,
        tracing=True,
    ) 
    SearchRunner(search_config).run()

    
    prompt_template = None
    if bundle.question_answering_prompt is not None:
        prompt_text = bundle.question_answering_prompt
        prompt_template = lambda: Template(prompt_text)
    
    runner_config = EvaluationRunnerConfig(
        search_results_path=os.path.join(
            iter_dir, 
            f"{args.top_k}_{sampled_index}_{sampled_index + 1}.json"
        ),
        dataset_type=DATASET_TYPE,
        qa_model=args.qa_model,
        judge_model=args.judge_model,
        qa_batch_size=args.qa_batch_size,
        judge_batch_size=args.judge_batch_size,
        api_config_path=(
            str(args.api_config_path)
            if args.api_config_path is not None
            else None
        ),
        prompt_template=prompt_template,
        traced_data_save_dir=traced_dir,
        tracing=True,
    )
    eval_results = EvaluationRunner(runner_config).run()

    metrics = {} 
    for eval_result in eval_results:
        for metric_name, metric_value in eval_result["metrics"].items():
            metrics[metric_name] = metric_value["value"] + metrics.get(metric_name, 0)
    for metric_name in metrics:
        metrics[metric_name] /= len(eval_results)

    return {
        "graph_path": os.path.join(
            iter_dir, 
            "traced_data", 
            user_id, 
            "graph_evaluation.json"
        ),
        "metrics": metrics,
        "num_queries": len(eval_results),
    }


def group_attributions(
    attributions: list[ErrorAttributionPrediction],
    exec_graph: ExecNetwork,
) -> tuple[
    dict[str, list[ErrorAttributionPrediction]], 
    list[ErrorAttributionPrediction],
    dict[str, str],
]:
    """Group attributions by optimizable prompt target.

    Args:
        attributions (`list[ErrorAttributionPrediction]`):
            Error-attribution predictions to group.
        exec_graph (`ExecNetwork`):
            Execution graph used to resolve attributed operation subgraphs.

    Returns:
        `tuple[
            dict[str, list[ErrorAttributionPrediction]], 
            list[ErrorAttributionPrediction], 
            dict[str, str],
        ]`:
            Attributions grouped by prompt target and attributions without a
            target, and a mapping from the full node identifier to the corresponding 
            field name.
    """
    grouped = {target: [] for target in PROMPT_TARGETS}
    unoptimizable = []

    full_node_id_to_name = {}
    for field_name, target_variable_name in PROMPT_TARGETS.items():
        target_versions = exec_graph.get_versions(target_variable_name)
        if target_versions:
            # It is abnormal to have multiple versions of the same target prompt variable.
            # For this case, we just throw a warning. 
            if len(target_versions) > 1:
                warnings.warn(
                    f"Multiple versions of the target variable '{target_variable_name}' "
                    "are found in the execution graph. The latest version is used."
                )
            full_node_id_to_name[target_versions[-1].full_node_id] = field_name
        else:
            raise ValueError(
                f"No target variable is found for '{field_name}' "
                "in the execution graph."
            )

    for attribution in attributions:
        # If the operation identifier is invalid, it returns an empty subgraph.
        subgraph = exec_graph.filter_by_operation(attribution.op_id) 
        target = None 
        for node in subgraph.nodes:
            candidate = full_node_id_to_name.get(node.full_node_id)
            if candidate is not None:
                target = candidate
                break
        if target is None:
            unoptimizable.append(attribution)
        else:
            grouped[target].append(attribution)
    return grouped, unoptimizable, full_node_id_to_name


def create_uniform_response_attributions(
    failed_cases: list[FailedQueryCase],
    exec_graph: ExecNetwork,
) -> list[ErrorAttributionPrediction]:
    """Attribute every failed case to its final response operation.

    Args:
        failed_cases (`list[FailedQueryCase]`):
            Failed query cases.
        exec_graph (`ExecNetwork`):
            Execution graph containing the question-answering operations.

    Returns:
        `list[ErrorAttributionPrediction]`:
            Response-error attributions for all failed cases.

    Raises:
        `ValueError`:
            If a failed case cannot be matched to a question-answering operation.
    """
    query_to_operation_id = {}
    response_operations = exec_graph.search_operations(
        name_pattern="^question-answering$",
        category="evaluation",
    )
    for operation in response_operations:
        operation_graph = exec_graph.filter_by_operation(operation.op_id)
        for node in operation_graph.nodes:
            if node.class_name == "query":
                query_to_operation_id[node.full_node_id] = operation.op_id

    attributions = []
    for failed_case in failed_cases:
        operation_id = query_to_operation_id.get(failed_case.query_full_node_id)
        if operation_id is None:
            raise ValueError(
                "No question-answering operation is found for failed query "
                f"'{failed_case.query_full_node_id}'."
            )
        attributions.append(
            ErrorAttributionPrediction(
                error_type="ResponseError",
                op_id=operation_id,
                reason=(
                    "The golden answer for this failed case is: "
                    f"{failed_case.golden_answer}. Therefore, the response "
                    "generated by this operation is incorrect."
                ),
            )
        )
    return attributions


def fill_missing_prompts_from_graphs(
    bundle: MemoryConfigBundle,
    exec_graph: ExecNetwork
) -> MemoryConfigBundle:
    """Fill missing values in the configuration bundle from the execution graph.
    
    Args:
        bundle (`MemoryConfigBundle`):
            The configuration bundle to be updated.
        exec_graph (`ExecNetwork`):
            The execution graph.

    Returns:
        `MemoryConfigBundle`:
            The updated configuration bundle.

    Raises:
        `RuntimeError`: 
            If unable to recover all target field value(s) from the execution graph.
    """
    updated = bundle.model_copy(deep=True)
    missing = [target for target in PROMPT_TARGETS if updated.get(target) is None]
    if not missing:
        return updated

    for target in missing:
        target_variable_name = PROMPT_TARGETS[target]
        # Get the latest version of the variable.
        target_versions = exec_graph.get_versions(target_variable_name)
        if target_versions: 
            runtime_variable = target_versions[-1]
            value = runtime_variable.value
            updated.update(target, value)

    still_missing = [target for target in PROMPT_TARGETS if updated.get(target) is None]
    if still_missing:
        raise RuntimeError(
            "It is unable to recover all the effective target values from the execution graph. "
            f"The following target fields are still missing: {', '.join(still_missing)}."
        )
    return updated


async def run(args: argparse.Namespace) -> None:
    """Run the automatic prompt optimization loop.
    
    Args:
        args (`argparse.Namespace`):
            Parsed command-line arguments.
    """
    last_iterations = find_last_n_iterations(
        args.optimization_dir, 
        return_iteration_numbers=True,
    )
    if args.rerun or last_iterations is None:
        start_iteration = 1  
        with open(args.config_path, "r", encoding="utf-8") as file:
            base_config = json.load(file)
        bundle_config = {
            "backend": "Mem0",
            "base_config": base_config,
        }
        bundle = MemoryConfigBundle.from_dict(bundle_config)
    else:
        # The last iteration directory is valid so we can load the bundle from it.
        start_iteration = last_iterations[-1][0] + 1
        bundle = MemoryConfigBundle.load(last_iterations[-1][1])
   
    num_iterations = args.sample_size
    if start_iteration > num_iterations:
        print(
            "The last iteration is already at or beyond "
            f"the sampled trajectory count ({num_iterations})."
        )
        return

    # Prepare the dataset in advance. 
    sampled_dataset_path = os.path.join(args.optimization_dir, SAMPLED_DATASET_FILE)
    load_dataset(
        args.dataset_path,
        dataset_type=DATASET_TYPE,
        sample_size=args.sample_size,
        seed=args.seed,
        save_path=sampled_dataset_path,
    )
    # The dataset paths used by subsequent scripts are all based on this one.  
    args.dataset_path = sampled_dataset_path

    memtrace = None
    if not args.disable_memtrace:
        memtrace_config = MemTraceConfig(
            model=args.attribution_model,
            max_trace_nodes=args.max_trace_nodes,
            stream=True, 
            temperature=1.0,
            context_window=args.context_window,
            max_iters=args.max_iters,
            api_config_path=args.api_config_path,
        )
        memtrace = MemTraceRunner(config=memtrace_config)

    # Create the optimizer. 
    # The optimization hisotry is stored in the optimizer.
    optimizer_config = OptimizerConfig(
        model=args.optimizer_model,
        stream=True,
        temperature=1.0,
        num_gradient_histories=args.num_gradient_histories,
        batch_size_for_summarization=args.batch_size_for_summarization,
        api_config_path=args.api_config_path,
    )
    optimizer = OptimizerRunner(config=optimizer_config)

    for iteration in range(start_iteration, num_iterations + 1):
        print(
            f"===== Running the {iteration}th trajectory-level prompt optimization iteration ====="
        )
        iter_dir_path = create_iter_dir(iteration, parent_dir=args.optimization_dir) 
        sampled_index = iteration - 1

        iter_summary = {} 

        # After creating the iteration directory, we set the save directory to this iteration directory. 
        bundle.update("save_dir", iter_dir_path)
        # For the Mem0 configuration, we also need to set the history database path to `None`. 
        bundle.update("history_db_path", None)
        
        pipeline_result = generate_execution_graph(
            args=args,
            iter_dir=iter_dir_path,
            sampled_index=sampled_index,
            bundle=bundle,
        )
        graph_path = pipeline_result["graph_path"]
        iter_summary.update(pipeline_result)

        # After running the whole pipeline, we load the corresponding execution graph.
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        exec_graph = ExecNetwork.import_graph(graph_data)

        # There are some prompts whose values are not available in the configuration bundle.
        # We need to fill them in the config bundle, so before the prompt optimization, 
        # the large language model knows the specific values of these prompts.
        effective_bundle = fill_missing_prompts_from_graphs(bundle, exec_graph)

        failed_cases_path = os.path.join(iter_dir_path, FAILED_CASES_FILE)
        failed_cases = collect_failed_queries(exec_graph)
        with open(failed_cases_path, "w", encoding="utf-8") as f:
            json.dump(
                [case.model_dump(mode="python") for case in failed_cases],
                f,
                indent=4,
                ensure_ascii=False,
            )

        attribution_path = os.path.join(iter_dir_path, ERROR_ATTRIBUTION_FILE)
        if failed_cases:
            if args.disable_memtrace:
                attributions = create_uniform_response_attributions(failed_cases, exec_graph)
                attribution_cost = {
                    "average_input_tokens": 0.0,
                    "average_output_tokens": 0.0,
                    "average_minutes": 0.0,
                }
            else:
                attributions, attribution_cost = await memtrace.arun(
                    failed_cases,
                    exec_graph,
                    batch_size=args.attribution_batch_size,
                    use_source_evidence_nodes=True,
                    memory_system="mem0",
                )
            with open(attribution_path, "w", encoding="utf-8") as f:
                json.dump(
                    [attribution.model_dump(mode="python") for attribution in attributions],
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
        else:
            attributions = []
            attribution_cost =  {
                "average_input_tokens": 0.0, 
                "average_output_tokens": 0.0,
                "average_minutes": 0.0,
            }
        iter_summary.update({"error_attribution_cost": attribution_cost})

        grouped, unoptimizable, full_node_id_to_name = group_attributions(
            [] if args.disable_memtrace else attributions,
            exec_graph,
        )
        if args.disable_memtrace:
            grouped = {
                target: attributions
                for target in PROMPT_TARGETS
            }
        name_to_full_node_id = {
            name: full_node_id
            for full_node_id, name in full_node_id_to_name.items()
        }
        iter_summary.update(
            {
                "num_optimizable_errors": sum(len(v) for v in grouped.values()),
                "num_unoptimizable_errors": len(unoptimizable),
            }
        )

        # Start the prompt optimization.
        # Note that we need to convert the variable name to the full node identifier
        # and filter out the targets with no optimizable errors.
        grouped_for_optimization = {
            name_to_full_node_id[key]: values
            for key, values in grouped.items()
            if len(values) > 0
        }
        if grouped_for_optimization:
            optimized_values, optimization_cost = await optimizer.arun(
                grouped_for_optimization,
                exec_graph,
                batch_size=args.optimization_batch_size,
                seed=args.seed + iteration, 
                save_folder=iter_dir_path,
            )
        else:
            optimized_values = {} 
            optimization_cost = {
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
        iter_summary.update({"optimization_cost": optimization_cost})

        
        for key, value in optimized_values.items():
            effective_bundle.update(full_node_id_to_name[key], value)
        
        # Save the updated configuration bundle.
        effective_bundle.save(iter_dir_path)
        bundle = effective_bundle.model_copy(deep=True)
        
        summary_path = os.path.join(iter_dir_path, SUMMARY_FILE)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                iter_summary,
                f,
                indent=4,
                ensure_ascii=False,
            )


def main() -> None:
    """Run the command-line entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()