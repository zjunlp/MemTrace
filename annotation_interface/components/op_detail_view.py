"""Detail subgraph renderer for edge and variable structures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from smartcomment.runtime.graph import RuntimeGraph

from data_engine.parser import GraphRecord
from error_attribution.trace_error_attribution import build_construction_subgraph
from utils.graph_render import (
    build_edge_render_geometry,
    build_ordered_layout,
    build_runtime_edge_render_attrs,
    build_runtime_node_render_attrs,
    color_with_alpha,
    edge_color_from_op_id,
    extract_selected_points,
    pick_interaction_payload,
)
from utils.runtime_graph import load_exec_network_and_root_variable
from utils.session_state import activate_related_view


@dataclass(slots=True)
class SubNode:
    node_id: str
    category: str
    label: str
    color: str
    x: float
    y: float
    hover: str
    detail: dict[str, Any]


@dataclass(slots=True)
class SubEdge:
    edge_id: str
    src: str
    dst: str
    op_id: str
    color: str
    hover: str
    detail: dict[str, Any]


def _runtime_node_key(full_node_id: str, category: str | None) -> str:
    """Build the frontend node key used by the detail subgraph renderer.

    Args:
        full_node_id (`str`):
            Runtime node identifier from smartcomment.
        category (`str | None`):
            Runtime node category.

    Returns:
        `str`:
            Prefixed node key in `msg|...` or `var|...` format.
    """
    normalized = (str(category or "").strip() or "unknown").lower()
    if "message" in normalized:
        return f"msg|{full_node_id}"
    return f"var|{full_node_id}"


def _build_runtime_subgraph_for_selected_pair(
    graph_path: str | Path | None,
) -> RuntimeGraph | None:
    """Build the construction subgraph for the selected macro edge.

    Args:
        graph_path (`str | Path | None`):
            Path to the source smartcomment graph JSON file.

    Returns:
        `RuntimeGraph | None`:
            Construction-stage runtime subgraph for the selected macro edge,
            or `None` when the runtime graph cannot be resolved.
    """
    pair_id = st.session_state.get("current_expanded_subgraph_id")
    if not pair_id or "->" not in pair_id or graph_path is None:
        return None

    source_id, _ = pair_id.split("->", 1)
    loaded = load_exec_network_and_root_variable(graph_path, source_id)
    if loaded is None:
        return None
    network, source_node = loaded
    session_graph = network.filter_by_session(source_node.session_id)
    return build_construction_subgraph(network, session_graph, source_node)


def _build_subgraph_from_runtime_graph(subgraph: RuntimeGraph) -> tuple[list[SubNode], list[SubEdge]]:
    """Convert one runtime subgraph into frontend node and edge view models.

    Args:
        subgraph (`RuntimeGraph`):
            Runtime graph returned by smartcomment.

    Returns:
        `tuple[list[SubNode], list[SubEdge]]`:
            Frontend node and edge models used by the detail view renderer.
    """
    if not subgraph.nodes:
        return [], []

    graph = nx.DiGraph()
    node_key_by_full_id: dict[str, str] = {}

    for node in subgraph.nodes:
        node_render = build_runtime_node_render_attrs(node)
        category = str(node_render["category"])
        node_key = _runtime_node_key(node.full_node_id, category)
        node_key_by_full_id[node.full_node_id] = node_key
        graph.add_node(
            node_key,
            category=category,
            label=str(node_render["label"]),
            color=str(node_render["color"]),
            hover=str(node_render["hover"]),
            detail={
                "node_id": node_key,
                "kind": "runtime_variable",
                "full_node_id": node.full_node_id,
                "full_name": node.full_name,
                "class_name": node.class_name,
                "category": node.category,
                "comment": node.comment,
                "value": node.value,
                "metadata": node.metadata,
            },
            created_at=str(node_render["created_at"]),
        )

    for edge in subgraph.edges:
        edge_render = build_runtime_edge_render_attrs(edge)
        src_node_id = node_key_by_full_id.get(edge.source_full_node_id)
        dst_node_id = node_key_by_full_id.get(edge.target_full_node_id)
        if not src_node_id or not dst_node_id:
            continue
        graph.add_edge(
            src_node_id,
            dst_node_id,
            id=edge.edge_id,
            hover=str(edge_render["hover"]),
            detail={
                "edge_id": edge.edge_id,
                "op_id": edge.op_id,
                "category": edge.category,
                "source_id": edge.source_full_node_id,
                "target_id": edge.target_full_node_id,
                "comment": edge.comment,
            },
        )

    created_at_by_node = {node_id: str(data.get("created_at") or "") for node_id, data in graph.nodes(data=True)}
    pos = build_ordered_layout(graph, created_at_by_node)

    nodes: list[SubNode] = []
    for node_id, coords in pos.items():
        node_data = graph.nodes[node_id]
        category = str(node_data["category"]).strip() or "unknown"
        nodes.append(
            SubNode(
                node_id=node_id,
                category=category,
                label=str(node_data["label"]),
                color=str(node_data["color"]),
                x=float(coords[0]),
                y=float(coords[1]),
                hover=str(node_data["hover"]),
                detail=dict(node_data["detail"]),
            )
        )

    edges: list[SubEdge] = []
    for u, v, data in graph.edges(data=True):
        edge_id = str(data["id"])
        edge_detail = dict(data["detail"])
        op_id = str(edge_detail.get("op_id") or "")
        edges.append(
            SubEdge(
                edge_id=edge_id,
                src=u,
                dst=v,
                op_id=op_id,
                color=edge_color_from_op_id(op_id),
                hover=str(data["hover"]),
                detail=edge_detail,
            )
        )

    return nodes, edges


def _add_edge_line_traces(
    fig: go.Figure,
    lines_by_color: dict[str, dict[str, list[float]]],
    *,
    alpha: float,
) -> None:
    """Add grouped edge line traces for one opacity bucket."""
    for color in sorted(lines_by_color):
        segment = lines_by_color[color]
        if not segment["x"]:
            continue
        fig.add_trace(
            go.Scattergl(
                x=segment["x"],
                y=segment["y"],
                mode="lines",
                line=dict(color=color_with_alpha(color, alpha), width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def render_op_detail_view(graph: GraphRecord, graph_path: str | Path | None = None) -> None:
    """Render the construction subgraph for the currently selected macro edge.

    Args:
        graph (`GraphRecord`):
            Parsed frontend graph record kept for renderer interface
            consistency with the main page.
        graph_path (`str | Path | None`, optional):
            Path to the source smartcomment graph JSON file.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    st.subheader("Edge & Variable Subgraph")

    pair_id = st.session_state.get("current_expanded_subgraph_id")
    if not pair_id or "->" not in pair_id:
        st.caption("After clicking the thick arrow in the main image, the corresponding edge and edge subgraph will be displayed here.")
        return

    runtime_subgraph = _build_runtime_subgraph_for_selected_pair(graph_path)
    if runtime_subgraph is None:
        st.caption("No construction subgraph is available for the selected edge.")
        return

    nodes, edges = _build_subgraph_from_runtime_graph(runtime_subgraph)
    if not nodes:
        st.caption("No construction subgraph is available for the selected edge.")
        return

    node_by_id = {n.node_id: n for n in nodes}
    all_x = [n.x for n in nodes]
    all_y = [n.y for n in nodes]
    margin = 1.0

    adjacency: dict[str, set[str]] = {n.node_id: set() for n in nodes}
    for edge in edges:
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)

    active_node = st.session_state.get("current_subgraph_highlight_var")
    active_edge = st.session_state.get("current_subgraph_highlight_edge")
    active_edge_nodes: set[str] = set()
    if active_edge:
        for edge in edges:
            if edge.edge_id == active_edge:
                active_edge_nodes = {edge.src, edge.dst}
                break
    active_set: set[str] = set()
    if active_node in adjacency:
        active_set = adjacency[active_node] | {active_node}

    category_to_color: dict[str, str] = {}
    for n in nodes:
        category_to_color[n.category] = n.color

    fig = go.Figure()
    lines_by_state: dict[str, dict[str, dict[str, list[float]]]] = {
        "active": {},
        "dim": {},
    }

    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []
    edge_custom: list[str] = []
    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_angles: list[float] = []
    arrow_colors: list[str] = []

    node_radius_gap = 0.45
    edge_hit_x: list[float] = []
    edge_hit_y: list[float] = []
    edge_hit_hover: list[str] = []
    edge_hit_custom: list[str] = []
    node_hit_x: list[float] = []
    node_hit_y: list[float] = []
    node_hit_hover: list[str] = []
    node_hit_custom: list[str] = []
    node_points = [(n.x, n.y) for n in nodes]

    for edge in edges:
        src = node_by_id[edge.src]
        dst = node_by_id[edge.dst]

        dx = dst.x - src.x
        dy = dst.y - src.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > node_radius_gap * 2.1:
            ux = dx / dist
            uy = dy / dist
            x0 = src.x + ux * node_radius_gap
            y0 = src.y + uy * node_radius_gap
            x1 = dst.x - ux * node_radius_gap
            y1 = dst.y - uy * node_radius_gap
        else:
            x0, y0, x1, y1 = src.x, src.y, dst.x, dst.y

        if active_edge:
            is_dim = edge.edge_id != active_edge
        else:
            is_dim = bool(active_set and not ({edge.src, edge.dst} & active_set))

        edge_color = edge.color
        edge_geometry = build_edge_render_geometry(x0, y0, x1, y1, node_points)
        state_key = "dim" if is_dim else "active"
        bucket = lines_by_state[state_key].setdefault(edge_color, {"x": [], "y": []})
        bucket["x"].extend([p[0] for p in edge_geometry.path_points] + [None])
        bucket["y"].extend([p[1] for p in edge_geometry.path_points] + [None])

        end_x, end_y = edge_geometry.arrow_end
        arrow_x.append(end_x)
        arrow_y.append(end_y)
        arrow_angles.append(edge_geometry.arrow_angle)
        arrow_colors.append(color_with_alpha(edge_color, 0.2 if is_dim else 1.0))

        if edge_geometry.midpoint is not None:
            mid_x, mid_y = edge_geometry.midpoint
            edge_mid_x.append(mid_x)
            edge_mid_y.append(mid_y)
            edge_hover.append(edge.hover)
            edge_custom.append(f"edge|{edge.edge_id}")

        for hx, hy in edge_geometry.hit_points:
            edge_hit_x.append(hx)
            edge_hit_y.append(hy)
            edge_hit_hover.append(edge.hover)
            edge_hit_custom.append(f"edge|{edge.edge_id}")

    for node in nodes:
        node_hit_x.append(node.x)
        node_hit_y.append(node.y)
        node_hit_hover.append(node.hover)
        node_hit_custom.append(node.node_id)

    _add_edge_line_traces(fig, lines_by_state["active"], alpha=1.0)
    _add_edge_line_traces(fig, lines_by_state["dim"], alpha=0.2)

    fig.add_trace(
        go.Scattergl(
            x=edge_mid_x,
            y=edge_mid_y,
            mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0.001)"),
            customdata=edge_custom,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=edge_hover,
            showlegend=False,
        )
    )

    if edge_hit_x:
        fig.add_trace(
            go.Scattergl(
                x=edge_hit_x,
                y=edge_hit_y,
                mode="markers",
                marker=dict(size=8, color="rgba(0,0,0,0.001)"),
                customdata=edge_hit_custom,
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=edge_hit_hover,
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_hit_x,
            y=node_hit_y,
            mode="markers",
            marker=dict(size=34, color="rgba(0,0,0,0.001)"),
            customdata=node_hit_custom,
            hovertext=node_hit_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="markers",
            marker=dict(size=5, color=arrow_colors, symbol="circle-dot", angle=arrow_angles),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    nodes_by_category: dict[str, list[SubNode]] = {}
    for n in nodes:
        nodes_by_category.setdefault(n.category, []).append(n)

    for category in sorted(nodes_by_category):
        cat_nodes = nodes_by_category[category]
        base_opacity = [
            1.0
            if (
                (not active_set and not active_edge)
                or (active_set and n.node_id in active_set)
                or (active_edge and n.node_id in active_edge_nodes)
            )
            else 0.2
            for n in cat_nodes
        ]
        fig.add_trace(
            go.Scattergl(
                x=[n.x for n in cat_nodes],
                y=[n.y for n in cat_nodes],
                mode="markers",
                marker=dict(
                    size=24,
                    color=[n.color for n in cat_nodes],
                    line=dict(color="#111827", width=1.2),
                    opacity=base_opacity,
                ),
                customdata=[n.node_id for n in cat_nodes],
                hovertext=[n.hover for n in cat_nodes],
                hovertemplate="%{hovertext}<extra></extra>",
                name=category,
                legendgroup=f"cat::{category}",
                showlegend=True,
            )
        )

    fig.update_layout(
        height=700,
        showlegend=bool(category_to_color),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(15,23,42,0.35)",
            borderwidth=1,
            font=dict(size=11),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[min(all_x) - margin, max(all_x) + margin]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[min(all_y) - margin, max(all_y) + margin], scaleanchor="x", scaleratio=1),
        clickmode="event+select",
        dragmode="pan",
        hovermode="closest",
        hoverlabel=dict(
            font_size=13,
        ),
        uirevision=graph.graph_id,
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"op_detail_plot_{graph.graph_id}",
        on_select="rerun",
        selection_mode=["points"],
    )

    points = extract_selected_points(event)
    if not points:
        return

    payload = pick_interaction_payload(points, preferred_kinds=["edge", "msg", "var", "node", "bind"])
    if payload and payload[0] == "edge":
        selected_edge_id = payload[1]
        current_edge = st.session_state.get("current_subgraph_highlight_edge")
        st.session_state["current_subgraph_highlight_edge"] = None if current_edge == selected_edge_id else selected_edge_id
        st.session_state["current_subgraph_highlight_var"] = None
        return

    if payload and payload[0] == "bind":
        return

    if payload and payload[0] in {"msg", "var", "node"}:
        selected_node = f"{payload[0]}|{payload[1]}"
    else:
        raw_custom = points[0].get("customdata")
        selected_node = str(raw_custom[0]) if isinstance(raw_custom, list) and raw_custom else str(raw_custom)
    current = st.session_state.get("current_subgraph_highlight_var")
    st.session_state["current_subgraph_highlight_var"] = None if current == selected_node else selected_node
    st.session_state["current_subgraph_highlight_edge"] = None

    if "|" in selected_node:
        root_node_id = selected_node.split("|", 1)[1]
        activate_related_view(root_node_id, return_target="main")
