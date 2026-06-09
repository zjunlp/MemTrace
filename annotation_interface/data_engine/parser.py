"""Parse backend graph schema and project it to frontend graph records."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from data_engine.qa_linkage import (
    LinkedEdgeRecord,
    LinkedNodeRecord,
    build_edge_indices,
    is_trace_query_node,
    resolve_query_linkage,
    should_skip_query_node,
    value_to_text,
)


@dataclass(slots=True)
class MessageRecord:
    message_id: str
    type: str
    text: str
    node_id: str | None = None
    create_at: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    messages: list[MessageRecord]


@dataclass(slots=True)
class QARecord:
    qa_id: str
    question: str
    golden_answer: str
    predicted_answer: str
    source_evidence_id: str | None
    is_correct: bool
    source_evidence_ids: list[str] = field(default_factory=list)
    source_evidence_texts: list[str] = field(default_factory=list)
    query_node_id: str | None = None
    query_full_name: str | None = None
    prediction_node_id: str | None = None


@dataclass(slots=True)
class GraphRecord:
    graph_id: str
    sessions: list[SessionRecord]
    messages: list[MessageRecord]
    qa_lists: list[QARecord]


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    """Validate that one parsed JSON value is a dictionary.

    Args:
        value (`Any`):
            Parsed JSON value to validate.
        name (`str`):
            Human-readable field path used in error messages.

    Returns:
        `dict[str, Any]`:
            The validated dictionary value.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    """Validate that one parsed JSON value is a list.

    Args:
        value (`Any`):
            Parsed JSON value to validate.
        name (`str`):
            Human-readable field path used in error messages.

    Returns:
        `list[Any]`:
            The validated list value.
    """
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _require_str(value: Any, name: str) -> str:
    """Validate that one parsed JSON value is a non-empty string.

    Args:
        value (`Any`):
            Parsed JSON value to validate.
        name (`str`):
            Human-readable field path used in error messages.

    Returns:
        `str`:
            The validated string value.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strip_trace_message_suffix(text: str) -> str:
    """Remove appended metadata blocks from message text.

    Args:
        text (`str`):
            Raw message text exported by smartcomment.

    Returns:
        `str`:
            Clean message text without the metadata suffix block.
    """
    marker = "\nBelow is this message's metadata:"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text.strip()


def _infer_trace_message_role(value: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Infer a conversational role for one message node.

    Args:
        value (`dict[str, Any]`):
            Parsed message value payload.
        metadata (`dict[str, Any]`):
            Node metadata attached to the message node.

    Returns:
        `str`:
            One of `"user"`, `"assistant"`, `"system"`, or `"unknown"`.
    """
    raw_role = metadata.get("speaker_role") or metadata.get("role") or value.get("role")
    if isinstance(raw_role, str):
        role = raw_role.strip().lower()
        if role in {"user", "assistant", "system"}:
            return role

    return "unknown"


def _parse_trace_message_value(value: Any) -> dict[str, Any]:
    """Parse one message value payload into a dictionary when possible.

    Args:
        value (`Any`):
            Raw message value stored on one trace node.

    Returns:
        `dict[str, Any]`:
            Parsed message payload dictionary, or an empty dictionary when the
            raw value is not dictionary-shaped.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _node_version_sort_key(node: dict[str, Any]) -> tuple[int, str, str]:
    """Return a stable ordering key for choosing the latest node version.

    Args:
        node (`dict[str, Any]`):
            Raw node object from the trace payload.

    Returns:
        `tuple[int, str, str]`:
            Comparison key ordered by numeric version, timestamp, and node ID.
    """
    raw_version = node.get("version")
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        version = -1
    return (
        version,
        str(node.get("created_at") or ""),
        str(node.get("full_node_id") or ""),
    )


def _resolve_latest_nodes_by_full_name(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve the newest node version for each `full_name`.

    Args:
        nodes (`list[dict[str, Any]]`):
            Raw node objects from the graph payload.

    Returns:
        `dict[str, dict[str, Any]]`:
            Mapping from `full_name` to the latest-version node object.
    """
    latest: dict[str, dict[str, Any]] = {}
    for node in nodes:
        full_name = str(node.get("full_name") or "").strip()
        if not full_name:
            continue
        prev = latest.get(full_name)
        if prev is None:
            latest[full_name] = node
            continue

        if _node_version_sort_key(node) >= _node_version_sort_key(prev):
            latest[full_name] = node
    return latest


def _collect_trace_qa_records(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> list[QARecord]:
    """Project QA records from raw smartcomment trace nodes and edges.

    Args:
        nodes (`list[dict[str, Any]]`):
            Raw node objects from the graph payload.
        edges (`list[dict[str, Any]]`):
            Raw edge objects from the graph payload.
        node_by_id (`dict[str, dict[str, Any]]`):
            Mapping from full node ID to raw node object.

    Returns:
        `list[QARecord]`:
            QA records derived from query, prediction, golden-answer, and judge nodes.
    """
    normalized_node_by_id = {
        node_id: LinkedNodeRecord(
            node_id=node_id,
            class_name=str(node.get("class_name") or ""),
            category=str(node.get("category") or ""),
            value=node.get("value"),
            full_name=str(node.get("full_name") or "") or None,
            metadata=node.get("metadata") if isinstance(node.get("metadata"), dict) else {},
        )
        for node_id, node in node_by_id.items()
    }
    normalized_edges = [
        LinkedEdgeRecord(
            source_id=str(edge.get("source_full_node_id") or ""),
            target_id=str(edge.get("target_full_node_id") or ""),
            op_id=str(edge.get("op_id") or "") or None,
        )
        for edge in edges
        if str(edge.get("source_full_node_id") or "").strip()
        and str(edge.get("target_full_node_id") or "").strip()
    ]
    edges_by_source, edges_by_target = build_edge_indices(normalized_edges)
    latest_by_full_name = _resolve_latest_nodes_by_full_name(nodes)
    query_nodes = [
        normalized_node_by_id[str(n.get("full_node_id"))]
        for n in nodes
        if is_trace_query_node(normalized_node_by_id[str(n.get("full_node_id"))])
    ]
    query_nodes.sort(key=lambda n: (str(node_by_id[n.node_id].get("created_at") or ""), n.node_id))

    qa_records: list[QARecord] = []
    for query_node in query_nodes:
        if should_skip_query_node(query_node):
            continue
        linkage = resolve_query_linkage(query_node, edges_by_source, edges_by_target, normalized_node_by_id)

        evidence_ids: list[str] = []
        evidence_texts: list[str] = []
        for full_name in linkage.source_evidence_full_names:
            latest_node = latest_by_full_name.get(full_name)
            if latest_node is None:
                continue
            evidence_id = latest_node.get("full_node_id")
            evidence_ids.append(evidence_id)
            evidence_texts.append(value_to_text(latest_node.get("value")))
        is_correct = str(linkage.judge_score_text) == "1.0"

        qa_records.append(
            QARecord(
                qa_id=f"qa::{linkage.query_node_id}",
                question=linkage.query_text,
                golden_answer=linkage.golden_text,
                predicted_answer=linkage.prediction_text,
                source_evidence_id=evidence_ids[0] if evidence_ids else None,
                is_correct=is_correct,
                source_evidence_ids=evidence_ids,
                source_evidence_texts=evidence_texts,
                query_node_id=linkage.query_node_id,
                query_full_name=linkage.query_full_name,
                prediction_node_id=linkage.prediction_node_id,
            )
        )

    return qa_records


def _parse_trace_exec_graph(obj: dict[str, Any], *, index: int) -> GraphRecord:
    """Parse one smartcomment execution graph into frontend records.

    Args:
        obj (`dict[str, Any]`):
            One trace graph object from the JSON payload.
        index (`int`):
            Zero-based graph index used in validation error messages.

    Returns:
        `GraphRecord`:
            Parsed graph record consumed by the Streamlit UI.
    """
    ctx = f"trace_graph[{index}]"
    graph_id = _require_str(obj.get("graph_id"), f"{ctx}.graph_id")

    data = _require_dict(obj.get("data"), f"{ctx}.data")
    raw_nodes = _require_list(data.get("nodes"), f"{ctx}.data.nodes")
    raw_edges = _require_list(data.get("edges") or [], f"{ctx}.data.edges")
    raw_sessions = _require_list(data.get("sessions") or [], f"{ctx}.data.sessions")

    session_ids: list[str] = []
    session_messages: dict[str, list[MessageRecord]] = {}
    for i, raw_session in enumerate(raw_sessions):
        s_obj = _require_dict(raw_session, f"{ctx}.data.sessions[{i}]")
        sid = _require_str(s_obj.get("session_id"), f"{ctx}.data.sessions[{i}].session_id")
        if sid in session_messages:
            continue
        session_ids.append(sid)
        session_messages[sid] = []

    nodes: list[dict[str, Any]] = []
    for i, raw_node in enumerate(raw_nodes):
        nctx = f"{ctx}.data.nodes[{i}]"
        node_obj = _require_dict(raw_node, nctx)
        node_obj = dict(node_obj)
        nodes.append(node_obj)
    nodes.sort(key=lambda n: (str(n.get("created_at") or ""), str(n.get("full_node_id") or "")))
    node_by_id = {str(n.get("full_node_id")): n for n in nodes}

    messages: list[MessageRecord] = []
    default_session_id = session_ids[0] if session_ids else None
    for i, node in enumerate(nodes):
        node_id = str(node.get("full_node_id"))
        category = str(node.get("category") or "variable").strip().lower()
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        raw_session_id = str(node.get("session_id") or "").strip()
        if raw_session_id:
            session_id = raw_session_id
        elif default_session_id is not None:
            session_id = default_session_id
        else:
            raise ValueError(
                f"{ctx}.data.nodes[{i}].session_id must be a non-empty string when no sessions are defined"
            )
        if session_id not in session_messages:
            session_messages[session_id] = []
            session_ids.append(session_id)

        if category == "message":
            raw_value = node.get("value")
            raw_text = value_to_text(raw_value)
            role = _infer_trace_message_role(
                _parse_trace_message_value(raw_value),
                metadata,
            )
            message = MessageRecord(
                message_id=node_id,
                node_id=str(node.get("node_id") or "") or None,
                type=role,
                text=_strip_trace_message_suffix(raw_text),
                create_at=str(node.get("created_at") or "") or None,
                session_id=session_id,
                metadata=metadata,
            )
            messages.append(message)
            session_messages[session_id].append(message)

    for sid in session_ids:
        session_messages.setdefault(sid, [])
        session_messages[sid].sort(key=lambda m: (m.create_at or "", m.message_id))
    sessions = [SessionRecord(session_id=sid, messages=session_messages[sid]) for sid in session_ids]

    qa_lists = _collect_trace_qa_records(nodes, raw_edges, node_by_id)

    messages.sort(key=lambda m: (m.create_at or "", m.message_id))
    return GraphRecord(
        graph_id=graph_id,
        sessions=sessions,
        messages=messages,
        qa_lists=qa_lists,
    )


def load_and_parse_json(json_path: str | Path) -> list[GraphRecord]:
    """Load one graph JSON file and parse it into frontend graph records.

    Args:
        json_path (`str | Path`):
            Path to the source graph JSON file.

    Returns:
        `list[GraphRecord]`:
            Parsed graph records loaded from the file.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    graph_obj = _require_dict(payload, "root")
    return [_parse_trace_exec_graph(graph_obj, index=0)]
