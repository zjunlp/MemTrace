# -*- coding: utf-8 -*-
"""Run MemTrace error attribution on a MemTraceBench split."""

import argparse
import asyncio
import copy
import json
import os
import random
from toolkits.bench_utils import load_memtracebench, MEMTRACEBENCH_SPLITS
from toolkits.memtrace_utils import MemTraceConfig, MemTraceRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        `argparse.Namespace`:
            The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run MemTrace error attribution on a MemTraceBench split.",
    )
    parser.add_argument(
        "dataset_dir",
        help="The MemTraceBench dataset directory containing the split directories.",
    )
    parser.add_argument(
        "split",
        choices=MEMTRACEBENCH_SPLITS,
        help="The single MemTraceBench split to attribute.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of execution graphs to sample. "
            "If omitted, all graphs in the split are used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample execution graphs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum number of attribution agents to run concurrently.",
    )
    parser.add_argument(
        "--provide-starting-nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed graph exploration with starting nodes.",
    )
    parser.add_argument(
        "--starting-nodes-type",
        choices=["pseudo_source_evidence", "source_evidence"],
        default="pseudo_source_evidence",
        help=(
            "Which starting nodes to begin from: 'pseudo_source_evidence' uses "
            "retrieved evidence while 'source_evidence' uses the real evidence."
        ),
    )
    parser.add_argument(
        "--use-system-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the split's memory-system prior knowledge to the instructions.",
    )
    parser.add_argument(
        "--exploration-strategy",
        choices=[
            "graph_search",
            "operation_block_search",
            "long_context",
        ],
        default="graph_search",
        help="The strategy used to explore the execution graph.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4.1-mini",
        help="OpenAI-compatible model name for the attribution agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the attribution agent.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to stream agent responses.",
    )
    parser.add_argument(
        "--max-trace-nodes",
        type=int,
        default=16,
        help="The maximum size of the agent's to-explore list.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum number of reasoning iterations per attribution case.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=272_000,
        help=(
            "The working context window size for the failure attribution agent. "
            "When the context exceeds this size, it is compressed automatically."
        ),
    )
    parser.add_argument(
        "--max-context-limit",
        type=int,
        default=1_000_000,
        help="The maximum context length allowed when compressing the working context.",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=3,
        help="Number of most recent messages kept uncompressed during compression.",
    )
    parser.add_argument(
        "--full-log-token-limit",
        type=int,
        default=600_000,
        help=(
            "Token budget for the flattened execution trace. "
            "It is truncated from the beginning so the latest operations can be kept. "
            "Note that this parameter is only used for the long-context baseline."
        ),
    )
    parser.add_argument(
        "--embedding-model-name",
        default="text-embedding-3-small",
        help="OpenAI-compatible embedding model name used for pseudo evidence retrieval.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=1536,
        help="The dimensionality of the embedding vectors.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="Base URL for the OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="API key for the OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for the embedding endpoint.",
    )
    parser.add_argument(
        "--retrieval-type",
        choices=["sparse", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval strategy used to select pseudo source-evidence starting points.",
    )
    parser.add_argument(
        "--starting-point-query-type",
        choices=[
            "query_with_golden_answer",
            "query_only",
            "query_with_prediction",
        ],
        default="query_with_golden_answer",
        help=(
            "The case fields used to construct the pseudo source-evidence "
            "starting-point retrieval query."
        ),
    )
    parser.add_argument(
        "--num-starting-points",
        type=int,
        default=None,
        help=(
            "Number of retrieved pseudo source-evidence starting points. "
            "If omitted, it defaults to half of the maximum trace nodes."
        ),
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=2,
        help="Candidate oversampling multiplier used by hybrid retrieval before fusion.",
    )
    parser.add_argument(
        "--studio-url",
        default=None,
        help="AgentScope Studio server URL.",
    )
    parser.add_argument(
        "--project",
        default="memtrace",
        help="AgentScope Studio project name.",
    )
    parser.add_argument(
        "--api-config-path",
        default=None,
        help="Path to an API config JSON with `api_keys` and `base_urls`.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory used to cache per-case attribution results. "
            "If omitted, caching is disabled."
        ),
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    """Run error attribution over a sampled MemTraceBench split.

    Args:
        args (`argparse.Namespace`):
            Parsed command-line arguments.

    Returns:
        `dict`:
            The full output payload.
    """
    assert os.path.splitext(args.output_path)[1] == ".json", (
        "Output path must be a JSON file. "
        f"However, '{args.output_path}' is provided instead."
    )

    # Load the execution graphs and their failed query cases. Non-memory-system
    # errors are dropped so only attributable cases remain.
    graphs, failed_cases = load_memtracebench(
        data_dir=args.dataset_dir,
        splits=args.split,
        filter_non_memory_errors=True,
    )
    assert failed_cases, f"No attributable cases are found in split '{args.split}'."

    # Sample whole execution graphs (and keep all of their cases) with a seed.
    if args.sample_size is not None:
        unique_graph_ids = list(dict.fromkeys(graph.graph_id for graph in graphs))
        assert 0 < args.sample_size <= len(unique_graph_ids), (
            f"`--sample-size` ({args.sample_size}) must be in "
            f"[1, {len(unique_graph_ids)}]."
        )
        sampled_graph_ids = set(
            random.Random(args.seed).sample(unique_graph_ids, args.sample_size)
        )
        selected = [
            (graph, case)
            for graph, case in zip(graphs, failed_cases)
            if graph.graph_id in sampled_graph_ids
        ]
        graphs = [graph for graph, _ in selected]
        failed_cases = [case for _, case in selected]

    config = MemTraceConfig(
        exploration_strategy=args.exploration_strategy,
        model=args.model_name,
        temperature=args.temperature,
        stream=args.stream,
        studio_url=args.studio_url,
        project=args.project,
        api_config_path=args.api_config_path,
        context_window=args.context_window,
        max_context_limit=args.max_context_limit,
        full_log_token_limit=args.full_log_token_limit,
        keep_recent=args.keep_recent,
        max_trace_nodes=args.max_trace_nodes,
        max_iters=args.max_iters,
        embedding_model_name=args.embedding_model_name,
        embedding_dimensions=args.embedding_dimensions,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        embedding_batch_size=args.embedding_batch_size,
        retrieval_type=args.retrieval_type,
        starting_point_query_type=args.starting_point_query_type,
        num_starting_points=args.num_starting_points,
        candidate_multiplier=args.candidate_multiplier,
        cache_dir=args.cache_dir,
    )
    runner = MemTraceRunner(config)

    # Map the starting-node arguments to the runner's attribution options.
    use_source_evidence_nodes = (
        args.provide_starting_nodes
        and args.starting_nodes_type == "source_evidence"
    )
    use_pseudo_source_evidence = (
        args.provide_starting_nodes
        and args.starting_nodes_type == "pseudo_source_evidence"
    )
    memory_system = args.split if args.use_system_prior else None

    predictions, error_attribution_cost = await runner.arun(
        failed_cases=failed_cases,
        graphs=graphs,
        batch_size=args.batch_size,
        use_source_evidence_nodes=use_source_evidence_nodes,
        use_pseudo_source_evidence=use_pseudo_source_evidence,
        memory_system=memory_system,
    )

    # Build the per-case results and the aggregate metrics. The accuracy is a
    # simple match between the prediction and the annotated label.
    results = []
    cases_by_graph = {}
    error_type_correct = 0
    op_id_correct = 0
    for prediction, case, graph in zip(predictions, failed_cases, graphs):
        annotation = case.metadata["annotation"]
        graph_name = os.path.join(
            args.dataset_dir,
            args.split,
            graph.metadata["memtracebench_graph_filename"],
        )
        is_error_type_correct = prediction.error_type == annotation["final_error_type"]
        is_op_id_correct = prediction.op_id == annotation["final_op_id"]
        error_type_correct += int(is_error_type_correct)
        op_id_correct += int(is_op_id_correct)
        cases_by_graph[graph_name] = cases_by_graph.get(graph_name, 0) + 1

        record = copy.deepcopy(annotation)
        record.update(
            {
                "graph_path": graph_name,
                "error_attribution_result": prediction.model_dump(),
                "is_error_type_correct": is_error_type_correct,
                "is_op_id_correct": is_op_id_correct,
            }
        )
        results.append(record)

    total = len(results)
    payload = {
        "split": args.split,
        "sampled_graph_paths": sorted(cases_by_graph),
        "results": results,
        "metrics": {
            "total_cases": total,
            "error_type_prediction_accuracy": error_type_correct / total,
            "operation_identifier_prediction_accuracy": op_id_correct / total,
            "cases_by_graph": cases_by_graph,
            "error_attribution_cost": error_attribution_cost,
        },
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )
    return payload


def main() -> None:
    """Run the command-line entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
