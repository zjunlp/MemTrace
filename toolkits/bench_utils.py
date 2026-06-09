import json
import os
from glob import glob
import warnings
from pydantic import (
    BaseModel, 
    Field, 
    JsonValue,
)
from smartcomment.runtime import ExecNetwork
from smartcomment.runtime.variable import RuntimeVariable
from smartcomment.runtime.errors import ExecNetworkKeyError 
from typing import Any, Iterable 


MEMORY_ERROR_TYPES = (
    "ExtractionError",
    "UpdateError",
    "DeletionError",
    "RetrievalError",
    "ResponseError",
)
MEMTRACEBENCH_SPLITS = ("rag", "mem0", "evermemos", "long_context")
MEMTRACEBENCH_HF_REPO_ID = "zjunlp/MemTraceBench"


class FailedQueryCase(BaseModel):
    """One query judged as incorrect in the execution graph."""

    query_full_node_id: str = Field(
        description="The full node identifier of the query node.",
    )
    query: str = Field( 
        description="The text of the query."
    )
    golden_answer: str = Field(
        description="The text of the golden answer.",
    )
    prediction: str = Field(
        description="The text of the question-answering model's prediction.",
    )
    source_evidence_full_node_ids: list[str] = Field(
        default_factory=list,
        description="The full node identifiers of the source evidence nodes.",
    )
    judge_response: str | None = Field(
        default=None,
        description="The text of the judge response.",
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="The metadata of the failed query case.",
    )


def find_source_evidence_nodes(
    graph: ExecNetwork,
    query_node: RuntimeVariable[Any],
) -> list[RuntimeVariable[Any]]:
    """Find source-evidence message nodes from a query node.

    Args:
        graph (`ExecNetwork`):
            The execution graph.
        query_node (`RuntimeVariable[Any]`):
            The query node.

    Returns:
        `list[RuntimeVariable[Any]]`:
            Source-evidence message nodes.
    """
    evidence_names = query_node.metadata.get("evidence") 
    if evidence_names is None:
        warnings.warn(
            f"Query node '{query_node.full_node_id}' has no source evidence metadata.",
            UserWarning,
        )
        return []

    nodes = []
    for evidence_name in evidence_names:
        try:
            nodes.append(
                graph.get_latest_variable(evidence_name)
            )
        except ExecNetworkKeyError:
            warnings.warn(
                f"The source evidence node '{evidence_name}' is not found in the graph. "
                "This node is skipped.", 
                UserWarning, 
            )
    return nodes 


def collect_failed_queries(graph: ExecNetwork) -> list[FailedQueryCase]:
    """Collect failed judged queries from an execution graph.

    Args:
        graph (`ExecNetwork`):
            Execution graph.

    Returns:
        `list[FailedQueryCase]`:
            A list of failed cases.
    """
    failed_cases = []
    judge_ops = graph.search_operations(
        name_pattern="^llm-judge$",
        category="evaluation",
    )

    # Each judge operation corresponds to one query.
    for judge_op in judge_ops:
        judge_subgraph = graph.filter_by_operation(judge_op.op_id)
        judge_score = judge_response = query_node = prediction = golden_answer = None
        for node in judge_subgraph.nodes:
            if node.class_name == "judge_score": 
                try:
                    judge_score = float(node.value)
                except ValueError as e: 
                    raise ValueError(
                        f"The value of judge score node '{node.full_node_id}' cannot be "
                        "converted to a float."
                    ) from e 
            elif node.class_name == "query":
                query_node = node
            elif node.class_name == "prediction":
                prediction = node.value 
            elif node.class_name == "golden_answers":
                golden_answer = node.value 
            elif node.class_name == "judge_response":
                judge_response = node.value 

        if (
            judge_score is None
            or query_node is None
            or prediction is None
            or golden_answer is None
            or judge_response is None
        ):
            warnings.warn(
                f"The judge operation '{judge_op.op_id}' is not complete. "
                "This operation is skipped.", 
                UserWarning,
            )
            continue 

        if judge_score != 0.0:
            continue

        evidence_nodes = find_source_evidence_nodes(graph, query_node)
        case = FailedQueryCase(
            query_full_node_id=query_node.full_node_id,
            query=query_node.value,
            golden_answer=golden_answer,
            prediction=prediction,
            source_evidence_full_node_ids=[
                node.full_node_id for node in evidence_nodes
            ],
            judge_response=judge_response,
            metadata=query_node.metadata,
        )
        failed_cases.append(case)

    return failed_cases


def load_memtracebench(
    data_dir: str,
    splits: Iterable[str] | str | None = None,
    filter_non_memory_errors: bool = True,
) -> tuple[list[ExecNetwork], list[FailedQueryCase]]:
    """Load the execution graphs and their failed query cases from MemTraceBench.

    Each JSON file under a split directory is imported as an execution graph.
    The failed query cases come from the curated annotations already stored in
    the data, rather than from re-deriving failed queries. The full annotation
    of each case is kept in the case metadata so it can be used for evaluation.

    The two returned lists have the same length and are paired by position: the
    graph at each position is the execution graph that the failed query case at
    the same position belongs to.

    If a requested split is not available locally, it is downloaded from the
    Hugging Face dataset repository into the data directory first.

    Args:
        data_dir (`str`):
            The dataset root directory. After loading, it contains the
            per-memory-system split directories.
        splits (`Iterable[str] | str | None`, optional):
            The split directories to load, e.g., ``"rag"`` or ``["rag", "mem0"]``. 
            If not provided, all splits are loaded.
        filter_non_memory_errors (`bool`, defaults to `True`):
            Whether to drop annotations whose error type is not a memory-system error 
            (e.g., annotation error or LLM-as-a-judge error). If it is disabled, 
            all annotations are kept.

    Returns:
        `tuple[list[ExecNetwork], list[FailedQueryCase]]`:
            The per-case execution graphs and the corresponding failed query
            cases, both of the same length.
    """
    if splits is None:
        splits = list(MEMTRACEBENCH_SPLITS)
    elif isinstance(splits, str):
        splits = [splits]
    else:
        splits = list(splits)

    # Download the splits that are not present locally yet.
    missing = [
        split
        for split in splits
        if not os.path.isdir(os.path.join(data_dir, split))
    ]
    if missing:
        os.makedirs(data_dir, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "`huggingface_hub` is required to download MemTraceBench from "
                "Hugging Face. Install it with `pip install huggingface_hub`."
            ) from e
        # Only the requested splits are fetched and placed directly under the
        # given data directory.
        snapshot_download(
            repo_id=MEMTRACEBENCH_HF_REPO_ID,
            repo_type="dataset",
            local_dir=data_dir,
            allow_patterns=[f"{split}/**" for split in missing],
        )

    graphs = []
    failed_cases = []
    for split in splits:
        # The dataset places each split directory directly under the given data directory.
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            warnings.warn(
                f"Split '{split}' is not found under '{data_dir}' and is skipped.",
                UserWarning,
            )
            continue

        for path in sorted(glob(os.path.join(split_dir, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            annotations = graph_data["data"]["annotations"]
            graph = ExecNetwork.import_graph(graph_data)
            for annotation in annotations:
                if (
                    filter_non_memory_errors
                    and annotation["final_error_type"] not in MEMORY_ERROR_TYPES
                ):
                    continue

                golden_answers = annotation["golden_answers"]
                if isinstance(golden_answers, list):
                    golden_answer = f"[{', '.join(golden_answers)}]"
                else:
                    golden_answer = golden_answers

                graphs.append(graph)
                failed_cases.append(
                    FailedQueryCase(
                        query_full_node_id=annotation["query_id"],
                        query=annotation["query"],
                        golden_answer=golden_answer,
                        prediction=annotation["prediction"],
                        source_evidence_full_node_ids=annotation["source_evidence_ids"],
                        metadata={"annotation": annotation},
                    )
                )

    return graphs, failed_cases
