# -*- coding: utf-8 -*-
"""Tool response helpers."""

from smartcomment.runtime.graph import RuntimeGraph
from smartcomment.runtime.operation import RuntimeOp
from typing import Literal


def _render_operation_subgraph(
    graph: RuntimeGraph,
    op: RuntimeOp,
    focus_full_node_id: str,
    *,
    render_format: Literal["markdown", "xml"] = "markdown",
    include_metadata: bool = False,
    include_variable_value: bool = True,
) -> str:
    """Render one operation subgraph based on the focus variable for the agent.

    Args:
        graph (`RuntimeGraph`):
            The operation subgraph.
        op (`RuntimeOp`):
            The operation being rendered.
        focus_full_node_id (`str`):
            The graph node currently being explored.
        render_format (`Literal["markdown", "xml"]`, defaults to `"markdown"`):
            Whether to render the canonical subgraph body as Markdown or XML.
        include_metadata (`bool`, defaults to `False`):
            Whether to include smartcomment metadata in the canonical body.
        include_variable_value (`bool`, defaults to `True`):
            Whether to include stored values for variable nodes in the canonical body.

    Returns:
        `str`:
            A readable operation-focused representation.
    """
    if focus_full_node_id not in graph:
        raise ValueError(
            f"The focus variable (full node identifier: `{focus_full_node_id}`) "
            "is not found in the operation subgraph. Please check the provided "
            "full node identifier.",
        )

    # Get a deterministically ordered copy of a runtime graph. 
    ordered_graph = RuntimeGraph(
        nodes=sorted(
            graph.nodes,
            key=lambda node: (node.created_at, node.full_node_id),
        ),
        edges=sorted(
            graph.edges,
            key=lambda edge: (edge.created_at, edge.edge_id),
        ),
        ops=sorted(
            graph.ops,
            key=lambda op: (op.created_at, op.op_id),
        ),
    )

    # Get the role of the focus variable.
    is_source = any(
        edge.source_full_node_id == focus_full_node_id for edge in graph.edges
    )
    is_target = any(
        edge.target_full_node_id == focus_full_node_id for edge in graph.edges
    )
    if is_source and is_target:
        role = (
            "The focus variable is an intermediate variable in this operation "
            "subgraph. It is produced as the output of one step, "
            "and then consumed as the input of another step."
        )
    elif is_source:
        role = (
            "The focus variable is an input to this operation. The operation "
            "reads it to produce downstream variables."
        )
    elif is_target:
        role = (
            "The focus variable is an output of this operation. It is produced "
            "by the operation."
        )
    else:
        role = (
            "The focus variable appears in this operation subgraph, but no direct "
            "input or output edge connects it to other variables."
        )

    # Get the direct successors of the focus variable.
    node_by_id = {node.full_node_id: node for node in graph.nodes}
    successor_ids = {
        edge.target_full_node_id
        for edge in graph.edges
        if edge.source_full_node_id == focus_full_node_id
    }
    successors = sorted(
        [
            node
            for full_node_id, node in node_by_id.items()
            if full_node_id in successor_ids
        ],
        key=lambda node: (node.created_at, node.full_node_id),
    )
    if successors: 
        successor_ids_str = ", ".join(
            f"`{successor.full_node_id}`" for successor in successors
        )
        successors_markdown = (
            f"There are {len(successors)} direct successor variables for the focus variable. "
            f"Their full node identifiers are: {successor_ids_str}."
        )
    else:
        successors_markdown = "There is no direct successor variable for the focus variable."

    # Get the leaf variables in the operation subgraph.
    leaves = sorted(
        ordered_graph.get_leaf_nodes(),
        key=lambda node: (node.created_at, node.full_node_id),
    )
    if leaves:
        leaf_ids_str = ", ".join(f"`{leaf.full_node_id}`" for leaf in leaves)
        leaves_markdown = (
            f"There are {len(leaves)} leaf variables in the operation subgraph. "
            f"Their full node identifiers are: {leaf_ids_str}."
        )
    else:
        leaves_markdown = "There is no leaf variable in the operation subgraph."

    if render_format == "markdown":
        body = ordered_graph.to_markdown(
            include_metadata=include_metadata,
            include_variable_value=include_variable_value,
        )
    elif render_format == "xml":
        body = ordered_graph.to_xml(
            include_metadata=include_metadata,
            include_variable_value=include_variable_value,
        )
    else:
        raise ValueError(
            f"The provided render format '{render_format}' is not supported. "
            "Only 'markdown' and 'xml' are supported.",
        )

    lines = [
        f"# Operation `{op.op_id}`",
        "",
        f"- Name: `{op.op_name or 'UNNAMED'}`",
        f"- Category: `{op.category}`",
        f"- Created At: `{op.created_at}`",
        *(
            [f"- Comment: {op.comment}"] if op.comment else []
        ), 
        f"- Focus Variable: `{focus_full_node_id}`",
        f"- Focus Role: {role}",
        "",
        "## Direct Successor Variables",
        successors_markdown,
        "",
        "## Leaf Variables In Operation Subgraph",
        leaves_markdown,
        "",
        "## Canonical Subgraph",
        body,
    ]
    return "\n".join(lines)


def _paginate_text(text: str, offset: int = 1, limit: int = 128_000) -> tuple[str, str]:
    """Slice text with one-based character offsets.

    Args:
        text (`str`):
            The full text to slice.
        offset (`int`, defaults to `1`):
            One-based start character offset. It must be greater than or equal
            to 1.
        limit (`int`, defaults to `128_000`):
            Maximum number of characters to return. It must be positive.

    Returns:
        `tuple[str, str]`:
            A tuple containing the sliced text and a status message describing the range.
    """
    if offset < 1:
        raise ValueError("`offset` must be greater than or equal to 1.")
    if limit < 1:
        raise ValueError("`limit` must be greater than or equal to 1.")

    total = len(text)
    start = offset - 1
    if start >= total:
        return (
            "",
            f"No content is available from the {offset}-th character. "
            f"The full content has {total} characters.",
        )

    end = min(start + limit, total)
    content = text[start:end]
    displayed_start = start + 1
    displayed_end = end

    if end < total:
        status = (
            f"Characters {displayed_start}-{displayed_end} of "
            f"{total} are returned. Content is truncated. Call this tool again with "
            f"`offset={displayed_end + 1}` to continue."
        )
    else:
        status = (
            f"Characters {displayed_start}-{displayed_end} of "
            f"{total} are returned. All content has been returned. No further read is "
            "needed."
        )
    return content, status

