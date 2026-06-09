"""Shared query-linkage helpers for parser and error attribution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(slots=True)
class LinkedNodeRecord:
    """Lightweight node view used by shared QA linkage resolution."""

    node_id: str
    class_name: str
    category: str
    value: Any
    full_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class LinkedEdgeRecord:
    """Lightweight edge view used by shared QA linkage resolution."""

    source_id: str
    target_id: str
    op_id: str | None = None


@dataclass(slots=True)
class QueryLinkageRecord:
    """Resolved linkage information for one query node."""

    query_node_id: str
    query_full_name: str | None
    query_text: str
    prediction_node_id: str | None
    prediction_text: str
    golden_node_id: str | None
    golden_text: str
    judge_score_text: str | None
    answer_op_id: str | None
    judge_op_id: str | None
    search_op_id: str | None
    source_evidence_full_names: list[str]


def value_to_text(value: Any) -> str:
    """Convert one value into frontend-safe display text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def is_trace_query_node(node: LinkedNodeRecord) -> bool:
    """Check whether one normalized node should be treated as a trace query."""
    return node.class_name == "query" and node.category in {"message", "message & query"}


def should_skip_query_node(node: LinkedNodeRecord) -> bool:
    """Check whether one query node should be skipped by QA extraction."""
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    marker = f"{node.node_id} {node.full_name or ''}"
    return metadata.get("question_type") == "adversarial" or "_abs" in marker


def build_edge_indices(
    edges: list[LinkedEdgeRecord],
) -> tuple[dict[str, list[LinkedEdgeRecord]], dict[str, list[LinkedEdgeRecord]]]:
    """Build source and target adjacency indices for normalized edges."""
    edges_by_source: dict[str, list[LinkedEdgeRecord]] = {}
    edges_by_target: dict[str, list[LinkedEdgeRecord]] = {}
    for edge in edges:
        edges_by_source.setdefault(edge.source_id, []).append(edge)
        edges_by_target.setdefault(edge.target_id, []).append(edge)
    return edges_by_source, edges_by_target


def resolve_query_linkage(
    query_node: LinkedNodeRecord,
    edges_by_source: dict[str, list[LinkedEdgeRecord]],
    edges_by_target: dict[str, list[LinkedEdgeRecord]],
    node_by_id: dict[str, LinkedNodeRecord],
) -> QueryLinkageRecord:
    """Resolve prediction, golden answer, judge score, and op linkage for one query."""
    metadata = query_node.metadata if isinstance(query_node.metadata, dict) else {}
    raw_evidence = metadata.get("evidence")
    source_evidence_full_names = [str(item) for item in raw_evidence] if isinstance(raw_evidence, list) else []

    outgoing_edges = edges_by_source.get(query_node.node_id, [])
    prediction_node_id: str | None = None
    prediction_text = "No Prediction"
    answer_op_id: str | None = None
    golden_node_id: str | None = None
    golden_text = "No Golden Answer"
    judge_op_id: str | None = None
    search_op_id: str | None = None
    for edge in outgoing_edges:
        target = node_by_id.get(edge.target_id)
        if target is None:
            continue

        if prediction_node_id is None and target.class_name == "prediction":
            prediction_node_id = target.node_id
            prediction_text = value_to_text(target.value)
            answer_op_id = edge.op_id

        if judge_op_id is None and target.class_name == "judge_response":
            judge_op_id = edge.op_id
            for candidate in edges_by_target.get(target.node_id, []):
                if candidate.source_id == query_node.node_id:
                    continue
                if candidate.op_id != judge_op_id:
                    continue
                source = node_by_id.get(candidate.source_id)
                if source is not None and source.class_name == "golden_answers":
                    golden_node_id = source.node_id
                    golden_text = value_to_text(source.value)
                    break

        if search_op_id is None and target.category == "memory_entry":
            search_op_id = edge.op_id

        if (
            prediction_node_id is not None
            and judge_op_id is not None
            and search_op_id is not None
        ):
            break

    judge_score_text: str | None = None
    queue = [query_node.node_id]
    visited = {query_node.node_id}
    while queue:
        current = queue.pop(0)
        for edge in edges_by_source.get(current, []):
            nxt = edge.target_id
            if not nxt or nxt in visited:
                continue
            visited.add(nxt)
            node = node_by_id.get(nxt)
            if node is not None and node.class_name == "judge_score":
                judge_score_text = value_to_text(node.value)
                queue.clear()
                break
            queue.append(nxt)

    return QueryLinkageRecord(
        query_node_id=query_node.node_id,
        query_full_name=query_node.full_name,
        query_text=value_to_text(query_node.value),
        prediction_node_id=prediction_node_id,
        prediction_text=prediction_text,
        golden_node_id=golden_node_id,
        golden_text=golden_text,
        judge_score_text=judge_score_text,
        answer_op_id=answer_op_id,
        judge_op_id=judge_op_id,
        search_op_id=search_op_id,
        source_evidence_full_names=source_evidence_full_names,
    )
