# -*- coding: utf-8 -*-
"""Evaluate the retrieval performance."""

import argparse
import asyncio
from agentscope.message import TextBlock
from agentscope.rag import Document, DocMetadata
from toolkits.bench_utils import load_memtracebench, MEMTRACEBENCH_SPLITS
from toolkits.memtrace_utils import DocumentRetriever
from typing import Literal


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        `argparse.Namespace`:
            The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k of the pseudo source-evidence starting points.",
    )
    parser.add_argument(
        "dataset_dir",
        help="The MemTraceBench dataset directory containing the split directories.",
    )
    parser.add_argument(
        "split",
        choices=MEMTRACEBENCH_SPLITS,
        help="The single MemTraceBench split to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="The number of retrieved starting points used to compute recall@k.",
    )
    parser.add_argument(
        "--retrieval-type",
        choices=["sparse", "dense", "hybrid"],
        default="hybrid",
        help="The retrieval strategy used to select starting points.",
    )
    parser.add_argument(
        "--starting-point-query-type",
        choices=[
            "query_with_golden_answer",
            "query_only",
            "query_with_prediction",
        ],
        default="query_with_golden_answer",
        help="The case fields used to construct the starting-point retrieval query.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=2,
        help="Candidate oversampling multiplier used by hybrid retrieval before fusion.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="text-embedding-3-small",
        help="OpenAI-compatible embedding model name used for retrieval.",
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
    return parser.parse_args()


def recall_at_k(
    references: list[list[str]],
    retrieved: list[list[str]],
    average: Literal["micro", "macro"] = "macro",
) -> float:
    """Compute the recall over all cases.

    Args:
        references (`list[list[str]]`):
            The relevant identifiers for each case.
        retrieved (`list[list[str]]`):
            The retrieved identifiers for each case.
        average (`Literal["micro", "macro"]`, defaults to `"macro"`):
            The averaging method. ``"macro"`` averages the per-case recall with
            equal weight per case. ``"micro"`` divides the total number of hits
            by the total number of relevant identifiers across all cases.

    Returns:
        `float`:
            The recall over the cases with relevant identifiers.
    """
    assert len(references) == len(retrieved), (
        "`references` and `retrieved` must have the same length."
    )
    if average == "macro":
        recalls = []
        for reference, retrieval in zip(references, retrieved):
            reference_set = set(reference)
            if not reference_set:
                continue
            recalls.append(len(reference_set & set(retrieval)) / len(reference_set))
        assert recalls, "No case with relevant identifiers to evaluate."
        return sum(recalls) / len(recalls)

    if average == "micro":
        total_hits = 0
        total_relevant = 0
        for reference, retrieval in zip(references, retrieved):
            reference_set = set(reference)
            total_hits += len(reference_set & set(retrieval))
            total_relevant += len(reference_set)
        assert total_relevant > 0, "No case with relevant identifiers to evaluate."
        return total_hits / total_relevant

    raise ValueError(
        f"`average` must be 'micro' or 'macro'. However, '{average}' is provided."
    )


async def run(args: argparse.Namespace) -> None:
    """Retrieve starting points per case and evaluate the retrieval performance.

    Args:
        args (`argparse.Namespace`):
            The parsed command-line arguments.
    """
    graphs, failed_cases = load_memtracebench(
        data_dir=args.dataset_dir,
        splits=args.split,
        filter_non_memory_errors=True,
    )
    assert failed_cases, f"No attributable cases are found in split '{args.split}'."

    datasets = []
    references = []
    retrieved = []
    for graph, case in zip(graphs, failed_cases):
        message_nodes = graph.search_variables(category="message")
        documents = [
            Document(
                metadata=DocMetadata(
                    content=TextBlock(type="text", text=node.value),
                    doc_id=node.full_node_id,
                    chunk_id=0,
                    total_chunks=1,
                ),
                id=node.full_node_id,
            )
            for node in message_nodes
        ]
        retriever = DocumentRetriever(
            embedding_model_name=args.embedding_model_name,
            embedding_dimensions=args.embedding_dimensions,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
            embedding_batch_size=args.embedding_batch_size,
            retrieval_type=args.retrieval_type,
            candidate_multiplier=args.candidate_multiplier,
        )
        await retriever.add_documents(documents)

        if args.starting_point_query_type == "query_only":
            query = case.query
        elif args.starting_point_query_type == "query_with_prediction":
            query = f"{case.query}\n{case.prediction}"
        else:
            query = f"{case.query}\n{case.golden_answer}"
        docs = await retriever.retrieve(query, limit=args.k)

        datasets.append(graph.metadata["memtracebench_dataset_source"])
        references.append(case.source_evidence_full_node_ids)
        retrieved.append([doc.metadata.doc_id for doc in docs])

    # Group cases by dataset source in a single pass, then report recall per
    # dataset source and overall, with case counts.
    grouped = {}
    for source, reference, retrieval in zip(datasets, references, retrieved):
        references_subset, retrieved_subset = grouped.setdefault(source, ([], []))
        references_subset.append(reference)
        retrieved_subset.append(retrieval)

    for dataset, (references_subset, retrieved_subset) in grouped.items():
        print(
            f"recall@{args.k} ({dataset}): "
            f"micro={recall_at_k(references_subset, retrieved_subset, average='micro'):.4f} "
            f"macro={recall_at_k(references_subset, retrieved_subset, average='macro'):.4f} "
            f"({len(references_subset)} cases)"
        )

    print(
        f"recall@{args.k} (overall): "
        f"micro={recall_at_k(references, retrieved, average='micro'):.4f} "
        f"macro={recall_at_k(references, retrieved, average='macro'):.4f} "
        f"({len(references)} cases)"
    )


def main() -> None:
    """Run the command-line entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
