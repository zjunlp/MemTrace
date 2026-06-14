"""The flatlog trace notebook class.

It manages flatlog-trace state, provides tool functions to the agent, and
returns normal ToolResponse errors when a tool call is malformed or fails.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agentscope.message import Msg, TextBlock
from agentscope.module import StateModule
from agentscope.tool import ToolResponse
from smartcomment.runtime.network import ExecNetwork
from smartcomment.runtime.operation import RuntimeEdge, RuntimeOp
from smartcomment.runtime.variable import RuntimeVariable

try:
    from .trace_index import TraceIndex, block_preview, match_to_dict
except ImportError:  # pragma: no cover - supports running this folder as scripts.
    from trace_index import TraceIndex, block_preview, match_to_dict


MAX_TOOL_CHARS = 128_000
ALLOWED_ERROR_TYPES = {
    "ExtractionError",
    "UpdateError",
    "DeletionError",
    "RetrievalError",
    "ResponseError",
}


def paginate_text(text: str, offset: int = 1, limit: int = 32_000) -> dict[str, Any]:
    """Return a bounded one-based character window from text.

    Args:
        text (`str`):
            The text to paginate.
        offset (`int`, defaults to `1`):
            One-based character offset where reading starts.
        limit (`int`, defaults to `32000`):
            Maximum number of characters to return. It is capped by
            `MAX_TOOL_CHARS`.

    Returns:
        `dict[str, Any]`:
            A dictionary with `content` and `status` fields.
    """
    limit = max(1, min(limit, MAX_TOOL_CHARS))
    if offset < 1:
        return {"content": "", "status": "Error: offset must be >= 1"}
    total = len(text)
    start = offset - 1
    if start >= total:
        return {"content": "", "status": f"No content from char {offset}; total chars={total}."}
    end = min(start + limit, total)
    status = f"Characters {start + 1}-{end} of {total} returned."
    if end < total:
        status += f" Call again with offset={end + 1}."
    return {"content": text[start:end], "status": status}


def source_evidence_nodes_to_initial_focus(
    nodes: list[RuntimeVariable[Any]] | None,
) -> list[dict[str, Any]]:
    """Convert MemTrace source-evidence nodes to notebook initial focus rows."""
    if not nodes:
        return []
    return [
        {
            "rank": rank,
            "created_at": node.created_at,
            "value": node.value,
        }
        for rank, node in enumerate(nodes, start=1)
    ]


def flatten_execution_graph(network: ExecNetwork) -> str:
    """Render an execution graph as a linear operation log."""
    ops = sorted(network.get_all_operations(), key=lambda op: (op.created_at, op.op_id))
    graph = network.to_runtime_graph()
    nodes_by_id = {node.full_node_id: node for node in graph.nodes}
    op_edges: dict[str, list[RuntimeEdge]] = defaultdict(list)
    op_node_ids: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        op_edges[edge.op_id].append(edge)
        op_node_ids[edge.op_id].add(edge.source_full_node_id)
        op_node_ids[edge.op_id].add(edge.target_full_node_id)

    rendered = []
    for op in ops:
        nodes = [
            nodes_by_id[node_id]
            for node_id in op_node_ids.get(op.op_id, set())
            if node_id in nodes_by_id
        ]
        rendered.append(_format_operation(op, nodes, op_edges.get(op.op_id, [])))
    return "\n\n".join(rendered) or "[NO RELATED OPERATIONS FOUND]"


def _format_operation(
    op: RuntimeOp,
    nodes: list[RuntimeVariable[Any]],
    edges: list[RuntimeEdge],
) -> str:
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    for edge in edges:
        out_deg[edge.source_full_node_id] += 1
        in_deg[edge.target_full_node_id] += 1

    nodes_by_id = {node.full_node_id: node for node in nodes}
    input_nodes: list[RuntimeVariable[Any]] = []
    output_nodes: list[RuntimeVariable[Any]] = []
    intermediate_nodes: list[RuntimeVariable[Any]] = []
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        node_in = in_deg.get(node_id, 0)
        node_out = out_deg.get(node_id, 0)
        if node_in == 0 and node_out > 0:
            input_nodes.append(node)
        elif node_out == 0 and node_in > 0:
            output_nodes.append(node)
        else:
            intermediate_nodes.append(node)

    parts = [
        f"### Operation {op.op_id}",
        f"created_at: {op.created_at}",
        f"op_name: {op.op_name}",
        f"category: {op.category}",
        _xml_field("comment", op.comment),
        _xml_field("metadata", json.dumps(_sanitize_identity_fields(op.metadata), ensure_ascii=False)),
    ]

    for header, section_nodes in (
        ("Input variables:", input_nodes),
        ("Output variables:", output_nodes),
        ("Intermediate variables:", intermediate_nodes),
    ):
        if not section_nodes:
            continue
        parts.extend(["", header])
        for node in section_nodes:
            lines = [
                "<variable>",
                _xml_field("class_name", node.class_name, indent="  "),
                _xml_field("category", node.category, indent="  "),
                _xml_field("created_at", node.created_at, indent="  "),
            ]
            if node.comment:
                lines.append(_xml_field("comment", node.comment, indent="  "))
            lines.append(_xml_field("value", node.value, indent="  "))
            lines.append("</variable>")
            parts.append("\n".join(lines))
    return "\n".join(parts)


def _xml_field(tag: str, value: Any, indent: str = "") -> str:
    text = "" if value is None else str(value)
    if "\n" not in text:
        return f"{indent}<{tag}>{_xml_escape(text)}</{tag}>"
    indented = "\n".join(f"{indent}  {_xml_escape(line)}" for line in text.splitlines())
    return f"{indent}<{tag}>\n{indented}\n{indent}</{tag}>"


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _sanitize_identity_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_identity_fields(item)
            for key, item in value.items()
            if key not in {"full_node_id", "node_id", "full_name", "name"}
        }
    if isinstance(value, list):
        return [_sanitize_identity_fields(item) for item in value]
    return value


class CCTraceNotebook(StateModule):
    """State notebook exposing edge-free flattened-trace tools to AgentScope."""

    hint_prefix: str = "<system-hint>"
    hint_suffix: str = "</system-hint>"
    not_started_hint: str = (
        "There is a flattened execution trace in the environment. Call "
        "`inspect_trace` first when the task requires trace inspection, direct "
        "trace exploration, finding or inspecting a specific operation, or "
        "following information across execution stages.\n\n"
    )
    started_hint: str = (
        "The flattened execution trace is active. Continue using only "
        "`inspect_trace`, `grep_log`, `view_log_window`, and "
        "`view_operation`.\n\n"
        "Use the current conversation and the previous tool results to decide "
        "what to inspect next. Avoid repeating the same searches when the "
        "needed information is already available in the viewed trace results."
    )

    description: str = (
        "The flattened execution-trace exploration tools. There is a "
        "predefined flattened execution trace in the environment. Activate "
        "these tools when you need to inspect operations, search trace text, "
        "view trace windows, summarize the trace, or answer any task that "
        "depends on the execution trace."
    )

    def __init__(
        self,
        flattened_trace_text: str,
        initial_ranked_ops: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the flatlog trace notebook.

        Args:
            flattened_trace_text (`str`):
                The edge-free operation trace rendered as a linear text log.
                It is parsed into operation blocks and kept as the only trace
                surface exposed to the agent.
            initial_ranked_ops (`list[dict[str, Any]] | None`, optional):
                Precomputed ranked starting operations derived before the
                agent begins reasoning.
        """
        super().__init__()
        self.index = TraceIndex(flattened_trace_text)
        self.initial_ranked_ops = initial_ranked_ops or []
        self.step = 0

    @classmethod
    def from_graph(
        cls,
        graph: ExecNetwork,
        source_evidence_nodes: list[RuntimeVariable[Any]] | None = None,
    ) -> "CCTraceNotebook":
        """Build a notebook from a MemTrace execution graph."""
        return cls(
            flattened_trace_text=flatten_execution_graph(graph),
            initial_ranked_ops=source_evidence_nodes_to_initial_focus(
                source_evidence_nodes,
            ),
        )

    def _tool_response(self, payload: dict[str, Any]) -> ToolResponse:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=json.dumps(payload, ensure_ascii=False),
                ),
            ],
            metadata=payload,
        )

    def _tool_error_response(
        self,
        tool: str,
        message: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResponse:
        """Create an error response for malformed or failed tool calls.

        Args:
            tool (`str`):
                Name of the tool that failed.
            message (`str`):
                User-facing error message for the agent.
            arguments (`dict[str, Any] | None`, optional):
                Tool arguments supplied by the agent, if available.

        Returns:
            `ToolResponse`:
                A normal AgentScope tool response containing an error payload.
        """
        payload: dict[str, Any] = {"tool": tool, "error": f"Error: {message}"}
        if arguments is not None:
            payload["arguments"] = arguments
        return self._tool_response(payload)

    async def inspect_trace(self) -> ToolResponse:
        """Inspect basic trace statistics and operation previews.

        Returns:
            `ToolResponse`:
                Operation count, log character count, and previews of the
                first and last operations in the flattened trace.
        """
        try:
            return self._tool_response(self.inspect_trace_result())
        except Exception as exc:  # pragma: no cover - defensive AgentScope guard.
            return self._tool_error_response("inspect_trace", f"{type(exc).__name__}: {exc}")

    def inspect_trace_result(self) -> dict[str, Any]:
        self.step += 1
        return {
            "operation_count": len(self.index.blocks),
            "log_char_count": len(self.index.flattened_trace_text),
            "first_ops": [block_preview(op) for op in self.index.blocks[:8]],
            "last_ops": [block_preview(op) for op in self.index.blocks[-8:]],
        }

    async def grep_log(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 40,
        surrounding_chars: int = 120,
    ) -> ToolResponse:
        """Search the flattened execution trace for a literal or regex pattern.

        Args:
            pattern (`str`):
                Literal text or regular expression to search for.
            regex (`bool`, defaults to `False`):
                Whether `pattern` should be interpreted as a regular
                expression. If `False`, the pattern is escaped and searched as
                literal text.
            case_sensitive (`bool`, defaults to `False`):
                Whether the search should be case-sensitive.
            max_matches (`int`, defaults to `40`):
                Maximum number of matches to return. It must be greater than
                or equal to 1.
            surrounding_chars (`int`, defaults to `120`):
                Number of context characters to include around each match. It
                must be non-negative.

        Returns:
            `ToolResponse`:
                Matched operation identifiers, snippets, and global character
                offsets that can be used with `view_log_window`.
        """
        args = {
            "pattern": pattern,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "max_matches": max_matches,
            "surrounding_chars": surrounding_chars,
        }
        if not isinstance(pattern, str) or not pattern:
            return self._tool_error_response("grep_log", "`pattern` must be a non-empty string.", args)
        if not isinstance(regex, bool):
            return self._tool_error_response("grep_log", "`regex` must be a boolean.", args)
        if not isinstance(case_sensitive, bool):
            return self._tool_error_response("grep_log", "`case_sensitive` must be a boolean.", args)
        if not isinstance(max_matches, int) or isinstance(max_matches, bool) or max_matches < 1:
            return self._tool_error_response("grep_log", "`max_matches` must be an integer >= 1.", args)
        if (
            not isinstance(surrounding_chars, int)
            or isinstance(surrounding_chars, bool)
            or surrounding_chars < 0
        ):
            return self._tool_error_response("grep_log", "`surrounding_chars` must be a non-negative integer.", args)
        try:
            return self._tool_response(
                self.grep_log_result(
                    pattern=pattern,
                    regex=regex,
                    case_sensitive=case_sensitive,
                    max_matches=max_matches,
                    surrounding_chars=surrounding_chars,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive AgentScope guard.
            return self._tool_error_response("grep_log", f"{type(exc).__name__}: {exc}", args)

    def grep_log_result(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 40,
        surrounding_chars: int = 120,
    ) -> dict[str, Any]:
        self.step += 1
        args = {
            "pattern": pattern,
            "regex": regex,
            "case_senstive": case_sensitive,
            "max_matches": max_matches,
            "surrounding_chars": surrounding_chars,
        }
        matches = self.index.grep(
            pattern=pattern,
            fields=["full"],
            regex=regex,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
            surrounding_chars=surrounding_chars,
        )
        matched_op_ids: list[str] = []
        for match in matches:
            if match.op_id not in matched_op_ids:
                matched_op_ids.append(match.op_id)
        result = {
            "pattern": pattern,
            "match_count": len(matches),
            "matched_op_ids": matched_op_ids,
            "matches": [match_to_dict(match) for match in matches],
        }
        return result

    async def view_log_window(self, offset: int, limit: int = 32_000) -> ToolResponse:
        """View a character window of the flattened execution trace.

        Args:
            offset (`int`):
                One-based global character offset in the flattened trace,
                usually taken from a `grep_log` match.
            limit (`int`, defaults to `32000`):
                Maximum number of characters to return. It must be greater
                than or equal to 1 and is capped by `MAX_TOOL_CHARS`.

        Returns:
            `ToolResponse`:
                A text window and pagination status.
        """
        args = {"offset": offset, "limit": limit}
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            return self._tool_error_response("view_log_window", "`offset` must be an integer >= 1.", args)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return self._tool_error_response("view_log_window", "`limit` must be an integer >= 1.", args)
        try:
            return self._tool_response(
                self.view_log_window_result(offset=offset, limit=limit),
            )
        except Exception as exc:  # pragma: no cover - defensive AgentScope guard.
            return self._tool_error_response("view_log_window", f"{type(exc).__name__}: {exc}", args)

    def view_log_window_result(self, offset: int, limit: int = 32_000) -> dict[str, Any]:
        self.step += 1
        page = paginate_text(self.index.flattened_trace_text, offset, limit)
        return page

    async def view_operation(
        self,
        op_id: str | None = None,
        op_index: int | None = None,
        offset: int = 1,
        limit: int = 32_000,
    ) -> ToolResponse:
        """View one operation block by operation id or index.

        Args:
            op_id (`str | None`, optional):
                Operation identifier to inspect. Use an id returned by
                `grep_log`, `inspect_trace`, or a previously viewed window.
            op_index (`int | None`, optional):
                Zero-based operation index to inspect. Use this when no
                operation id is known yet.
            offset (`int`, defaults to `1`):
                One-based character offset inside the operation block.
            limit (`int`, defaults to `32000`):
                Maximum number of operation-block characters to return. It
                must be greater than or equal to 1 and is capped by
                `MAX_TOOL_CHARS`.

        Returns:
            `ToolResponse`:
                Operation metadata, content window, and pagination status.
        """
        args = {"op_id": op_id, "op_index": op_index, "offset": offset, "limit": limit}
        if op_id is not None and not isinstance(op_id, str):
            return self._tool_error_response("view_operation", "`op_id` must be a string when provided.", args)
        if op_index is not None and (not isinstance(op_index, int) or isinstance(op_index, bool)):
            return self._tool_error_response("view_operation", "`op_index` must be an integer when provided.", args)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            return self._tool_error_response("view_operation", "`offset` must be an integer >= 1.", args)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return self._tool_error_response("view_operation", "`limit` must be an integer >= 1.", args)
        try:
            return self._tool_response(
                self.view_operation_result(
                    op_id=op_id,
                    op_index=op_index,
                    offset=offset,
                    limit=limit,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive AgentScope guard.
            return self._tool_error_response("view_operation", f"{type(exc).__name__}: {exc}", args)

    def view_operation_result(
        self,
        op_id: str | None = None,
        op_index: int | None = None,
        offset: int = 1,
        limit: int = 32_000,
    ) -> dict[str, Any]:
        self.step += 1
        block = self._resolve_operation(op_id=op_id, op_index=op_index)
        if isinstance(block, dict):
            return block
        op_id = block.op_id
        page = paginate_text(block.full_text, offset, limit)
        return {**block_preview(block), **page}

    def finish_attribution(self, error_type: str, op_id: str, reason: str) -> dict[str, Any]:
        """Validate the final structured attribution.

        Args:
            error_type (`str`):
                Predicted memory-system error type.
            op_id (`str`):
                Predicted earliest decisive faulty operation id.
            reason (`str`):
                Trace-grounded reason for the attribution.

        Returns:
            `dict[str, Any]`:
                Final attribution payload, optionally with an `error` field if
                validation rejects the prediction.
        """
        self.step += 1
        result = {"error_type": error_type, "op_id": op_id, "reason": reason}
        validation_error = self._validate_final(error_type, op_id, reason)
        if validation_error:
            result["error"] = validation_error
        return result

    def get_current_hint_text(self) -> str:
        """Build the current trace-investigation hint.

        Returns:
            `str`:
                Hint text wrapped by `<system-hint></system-hint>`. The hint
                has two fixed states, before and after the first tool call.
        """
        hint = self.started_hint if self._has_started_processing() else self.not_started_hint
        initial_focus = self._build_initial_focus_text()
        if initial_focus:
            hint = f"{hint}\n\n{initial_focus}"
        return f"{self.hint_prefix}\n{hint}\n{self.hint_suffix}"

    async def get_current_hint(self) -> Msg:
        """Get the current trace hint message.

        Returns:
            `Msg`:
                AgentScope user-role message containing the current notebook
                hint.
        """
        return Msg("user", self.get_current_hint_text(), "user")

    def list_tools(self) -> list[Callable[..., Any]]:
        """List all AgentScope tool functions exposed by this notebook.

        Returns:
            `list[Callable[..., Any]]`:
                Tool functions registered into the agent toolkit.
        """
        return [
            self.inspect_trace,
            self.grep_log,
            self.view_log_window,
            self.view_operation,
        ]

    def _has_started_processing(self) -> bool:
        """Return whether at least one trace tool/finalization step ran.

        Returns:
            `bool`:
                `True` after the notebook has recorded at least one tool or
                finalization step; `False` before processing starts.
        """
        return self.step > 0

    def _build_initial_focus_text(self) -> str:
        lines = []
        if self.initial_ranked_ops:
            lines.append("Source evidence:")
            for item in self.initial_ranked_ops[:8]:
                lines.append(
                    f"### Source Evidence {item.get('rank')}\n"
                    f"created_at: {item.get('created_at')}\n"
                    f"value: {item.get('value')}"
                )
        return "\n".join(lines)

    def _resolve_operation(self, op_id: str | None, op_index: int | None) -> Any:
        if op_id:
            block = self.index.get_op(op_id)
            return block if block is not None else {"error": f"op_id not found: {op_id}"}
        if op_index is None:
            return {"error": "Provide either op_id or op_index."}
        if op_index < 0 or op_index >= len(self.index.blocks):
            return {"error": f"op_index out of range: {op_index}; operation_count={len(self.index.blocks)}"}
        return self.index.blocks[op_index]

    def _validate_final(self, error_type: str, op_id: str, reason: str) -> str:
        if error_type not in ALLOWED_ERROR_TYPES:
            return f"Invalid error_type {error_type!r}. Use one of {sorted(ALLOWED_ERROR_TYPES)}."
        if not op_id:
            return "Missing op_id. You must choose an operation id from the viewed trace."
        block = self.index.get_op(op_id)
        if block is None:
            return f"op_id not found in the flattened trace: {op_id}"
        if not reason or len(reason.strip()) < 40:
            return "Reason is too short. Explain the local fault, rescue condition, and why earlier candidates are not decisive."
        return ""
