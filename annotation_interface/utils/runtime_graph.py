"""Shared smartcomment runtime graph loaders and cached graph-level helpers."""

from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from typing import Any

import streamlit as st
from smartcomment.runtime.network import ExecNetwork


def _runtime_edge_payload(
    edge: Any,
    *,
    source_key: str,
    target_key: str,
    stringify_created_at: bool,
) -> dict[str, Any]:
    """Serialize one runtime edge into a frontend-friendly payload."""
    payload = {
        "edge_id": edge.edge_id,
        "op_id": edge.op_id,
        "category": edge.category,
        "comment": edge.comment,
        source_key: edge.source_full_node_id,
        target_key: edge.target_full_node_id,
    }
    payload["created_at"] = (
        str(getattr(edge, "created_at", "") or "")
        if stringify_created_at
        else edge.created_at
    )
    return payload


@st.cache_resource(show_spinner=False)
def load_exec_network(graph_path: str) -> ExecNetwork:
    """Load and cache one smartcomment execution network from disk.

    Args:
        graph_path (`str`):
            Path to the source smartcomment graph JSON file.

    Returns:
        `ExecNetwork`:
            Cached execution network imported from the graph file.
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    return ExecNetwork.import_graph(graph_data)


@st.cache_data(show_spinner=False)
def load_all_graph_op_edge_ids(graph_path: str) -> tuple[set[str], set[str]]:
    """Load all operation IDs and edge IDs from one smartcomment graph file.

    Args:
        graph_path (`str`):
            Path to the source smartcomment graph JSON file.

    Returns:
        `tuple[set[str], set[str]]`:
            Sets of all non-empty `op_id` and `edge_id` values in the graph.
    """
    network = load_exec_network(graph_path)
    op_ids: set[str] = set()
    edge_ids: set[str] = set()
    for edge in network.get_all_edges():
        op_id = str(edge.op_id or "").strip()
        edge_id = str(edge.edge_id or "").strip()
        if op_id:
            op_ids.add(op_id)
        if edge_id:
            edge_ids.add(edge_id)
    return op_ids, edge_ids


def load_exec_network_and_root_variable(
    graph_path: str | Path | None,
    root_node_id: str,
) -> tuple[ExecNetwork, Any] | None:
    """Load one runtime graph and resolve one root variable from it.

    Args:
        graph_path (`str | Path | None`):
            Path to the source smartcomment graph JSON file.
        root_node_id (`str`):
            Full node ID of the runtime variable that should be resolved.

    Returns:
        `tuple[ExecNetwork, Any] | None`:
            Loaded execution network and resolved runtime variable, or `None`
            when the path is missing, unreadable, or the variable is absent.
    """
    if graph_path is None:
        return None

    resolved_graph_path = Path(graph_path).resolve()
    if not resolved_graph_path.exists():
        return None

    try:
        network = load_exec_network(str(resolved_graph_path))
    except (OSError, json.JSONDecodeError, ValueError):
        return None

    try:
        root_node = network.get_variable(root_node_id)
    except KeyError:
        return None

    if root_node is None:
        return None

    return network, root_node


def runtime_edges_to_payloads(subgraph: Any) -> list[dict[str, Any]]:
    """Convert runtime edges into lightweight frontend payload dictionaries.

    Args:
        subgraph (`Any`):
            Runtime graph object returned by smartcomment.

    Returns:
        `list[dict[str, Any]]`:
            Serialized edge payloads that preserve IDs, endpoints, category,
            comment, and timestamp fields.
    """
    return [
        _runtime_edge_payload(
            edge,
            source_key="source_id",
            target_key="target_id",
            stringify_created_at=True,
        )
        for edge in subgraph.edges
    ]


def runtime_graph_to_payload(subgraph: Any) -> dict[str, Any]:
    """Convert one runtime subgraph into the serialized frontend graph payload.

    Args:
        subgraph (`Any`):
            Runtime graph object returned by smartcomment.

    Returns:
        `dict[str, Any]`:
            Frontend payload with `nodes` and `edges` lists.
    """
    nodes = []
    for node in subgraph.nodes:
        metadata = node.metadata or {}
        nodes.append(
            {
                "id": node.full_node_id,
                "node_id": node.node_id,
                "full_name": node.full_name,
                "class_name": node.class_name,
                "category": node.category,
                "type": metadata.get("type"),
                "created_at": node.created_at,
                "value": node.value,
                "comment": node.comment,
                "metadata": metadata,
            }
        )

    edges = []
    for edge in subgraph.edges:
        edges.append(
            _runtime_edge_payload(
                edge,
                source_key="source",
                target_key="target",
                stringify_created_at=False,
            )
        )

    return {"nodes": nodes, "edges": edges}


def normalize_candidate_op_ids(candidate_op_ids: Any) -> list[str]:
    """Normalize one mixed candidate-op input into a unique ordered string list.

    Args:
        candidate_op_ids (`Any`):
            Candidate operation IDs stored as a list-like object, JSON string,
            Python repr string, or whitespace/comma separated text.

    Returns:
        `list[str]`:
            Unique non-empty operation IDs in first-seen order.
    """
    normalized_ops: list[str] = []

    def _add_op(op_value: Any) -> None:
        op_id = str(op_value or "").strip()
        if op_id and op_id not in normalized_ops:
            normalized_ops.append(op_id)

    if isinstance(candidate_op_ids, (list, tuple, set)):
        for item in candidate_op_ids:
            _add_op(item)
        return normalized_ops

    raw = str(candidate_op_ids or "").strip()
    if not raw:
        return normalized_ops

    parsed: Any = None
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = None

    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            _add_op(item)
        return normalized_ops

    for token in re.split(r"[\s,]+", raw):
        _add_op(token)
    return normalized_ops


def build_op_edge_index_markdown(
    edge_payloads: list[dict[str, Any]],
    candidate_op_ids: Any = None,
) -> str:
    """Build grouped `op_id -> edge_id` markdown for one graph edge collection.

    Args:
        edge_payloads (`list[dict[str, Any]]`):
            Serialized edge payloads that contain `op_id` and `edge_id`.
        candidate_op_ids (`Any`, optional):
            Optional candidate operation ID input that should be merged into the
            displayed operation list even if no edge payload currently matches.

    Returns:
        `str`:
            Markdown bullet list describing operation IDs and their edge IDs,
            or `"(none)"` when nothing is available.
    """
    op_to_edges: dict[str, list[str]] = {}
    for edge in edge_payloads:
        op_id = str(edge.get("op_id") or "").strip()
        edge_id = str(edge.get("edge_id") or "").strip()
        if not op_id:
            continue
        if op_id not in op_to_edges:
            op_to_edges[op_id] = []
        if edge_id and edge_id not in op_to_edges[op_id]:
            op_to_edges[op_id].append(edge_id)

    ordered_ops = list(normalize_candidate_op_ids(candidate_op_ids))
    for op_id in op_to_edges:
        if op_id not in ordered_ops:
            ordered_ops.append(op_id)

    if not ordered_ops:
        return "(none)"

    lines: list[str] = []
    for op_id in ordered_ops:
        lines.append(f"- op_id: {op_id}")
        edge_ids = op_to_edges.get(op_id, [])
        if not edge_ids:
            lines.append("    - edge_id: (none)")
            continue
        for edge_id in sorted(edge_ids):
            lines.append(f"    - edge_id: {edge_id}")
    return "\n".join(lines)
