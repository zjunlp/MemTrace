"""Dataset indexing helpers for graph discovery and lookup."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from data_engine.loader import load_graph_by_name, load_graph_records


def dataset_index_path(dataset_path: Path) -> Path:
    """Return the cached index path for one dataset location."""
    abs_path = dataset_path.resolve().as_posix()
    path_hash = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:12]
    app_dir = Path(__file__).resolve().parent.parent
    return app_dir / f"dataset_index_{path_hash}.json"


def dataset_graph_files(dataset_path: Path) -> list[Path]:
    """Return candidate graph JSON files for one dataset path."""
    if dataset_path.is_file():
        return [dataset_path]
    if dataset_path.is_dir():
        direct_files = sorted(
            p for p in dataset_path.glob("*.json")
            if p.is_file() and not p.name.startswith("dataset_index_")
        )
        if direct_files:
            return direct_files
        return sorted(
            p for p in dataset_path.rglob("*.json")
            if p.is_file() and not p.name.startswith("dataset_index_")
        )
    return []


def build_dataset_index_file(dataset_path: str, index_path: Path) -> None:
    """Build and write the lightweight dataset index used by the app."""
    path = Path(dataset_path)
    if not path.exists():
        print(f"Error: Directory or file {dataset_path} does not exist.")
        return

    graph_files = dataset_graph_files(path)
    if not graph_files:
        print(f"Invalid dataset path: {dataset_path}")
        return

    loaded_entries: list[dict[str, Any]] = []
    print(f"Found {len(graph_files)} JSON files. Building index...")

    for i, graph_file in enumerate(graph_files, start=1):
        print(f"[{i}/{len(graph_files)}] Processing {graph_file.name}...")
        try:
            file_graphs = load_graph_records(graph_file)
            for idx, graph in enumerate(file_graphs, start=1):
                graph_name = graph_file.name if idx == 1 else f"{graph_file.name}#{idx}"
                wrong_qas = [
                    {
                        "qa_id": qa.qa_id,
                        "query_full_name": qa.query_full_name or "(Unknown)",
                    }
                    for qa in graph.qa_lists
                    if not qa.is_correct
                ]
                loaded_entries.append(
                    {
                        "graph_name": graph_name,
                        "graph_path": str(graph_file.absolute()),
                        "wrong_qas": wrong_qas,
                    }
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"  -> Error parsing {graph_file.name}: {exc}")

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(loaded_entries, f, ensure_ascii=False, indent=2)
    print(f"\nIndex built successfully. Saved to {index_path}")


def is_dataset_index_stale(dataset_path: Path, index_path: Path) -> bool:
    """Return whether the on-disk dataset index should be rebuilt."""
    if not index_path.exists():
        return True

    index_mtime = index_path.stat().st_mtime
    if dataset_path.is_file():
        return dataset_path.stat().st_mtime > index_mtime
    if dataset_path.is_dir():
        for graph_file in dataset_graph_files(dataset_path):
            if graph_file.stat().st_mtime > index_mtime:
                return True
    return False


def load_dataset_index_file(index_path: Path) -> list[dict[str, Any]] | None:
    """Load one dataset index file from disk."""
    try:
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

def build_data_version_token(data_path: Path) -> str:
    """Build a stable cache token from graph file metadata."""
    if data_path.is_file():
        return str(data_path.stat().st_mtime_ns)

    if data_path.is_dir():
        signature_chunks: list[str] = []
        for graph_file in dataset_graph_files(data_path):
            stat = graph_file.stat()
            signature_chunks.append(
                f"{graph_file.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
            )
        joined = "|".join(signature_chunks)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    return "missing"
