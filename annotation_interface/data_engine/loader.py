"""Data loading helpers for graph protocol files."""

from __future__ import annotations

from pathlib import Path

from data_engine.parser import GraphRecord, load_and_parse_json


def load_graph_records(json_path: str | Path) -> list[GraphRecord]:
    """Load parsed graph records from one source JSON file.

    Args:
        json_path (`str | Path`):
            Path to the source graph JSON file.

    Returns:
        `list[GraphRecord]`:
            Parsed graph records loaded from the target file.
    """
    return load_and_parse_json(Path(json_path))


def load_graph_by_name(graph_path: str | Path, target_graph_name: str) -> GraphRecord | None:
    """Load one parsed graph record from a source JSON file by display name.

    Args:
        graph_path (`str | Path`):
            Path to the source graph JSON file.
        target_graph_name (`str`):
            Display name shown in the selector, including optional `#<index>`
            suffix for multi-graph files.

    Returns:
        `GraphRecord | None`:
            The parsed graph record matching `target_graph_name`, or `None`
            when no matching graph exists in the file.
    """
    graph_path = Path(graph_path)
    for idx, graph in enumerate(load_graph_records(graph_path), start=1):
        graph_name = graph_path.name if idx == 1 else f"{graph_path.name}#{idx}"
        if graph_name == target_graph_name:
            graph.graph_id = graph_name
            return graph
    return None
