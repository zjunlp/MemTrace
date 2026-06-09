"""Related-variable view rendered as an overlay-style replacement for main/subgraph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from smartcomment.runtime.graph import RuntimeGraph

from data_engine.parser import GraphRecord
from utils.session_state import (
    DEFAULT_RELATED_VIEW_BFS_MODE,
    close_related_view,
    ensure_related_view_stack,
    push_related_view_node,
    step_back_related_view,
)

from utils.graph_render import (
    build_edge_render_geometry,
    build_ordered_layout,
    build_runtime_edge_render_attrs,
    build_runtime_node_render_attrs,
    extract_selected_points,
    pick_clicked_node_id,
)
from utils.runtime_graph import (
    build_op_edge_index_markdown,
    load_exec_network_and_root_variable,
    runtime_edges_to_payloads,
)


@dataclass(slots=True)
class RelNode:
    node_id: str
    category: str
    label: str
    color: str
    hover: str


VALID_BFS_MODES = {"backward", "forward", "both"}
BFS_MODE_LABEL = {
    "backward": "Backward BFS",
    "forward": "Forward BFS",
    "both": "Bidirectional BFS",
}


def _build_runtime_related_subgraph(
    graph_path: str | Path | None,
    root_node_id: str,
    bfs_mode: str,
) -> RuntimeGraph | None:
    """Build a one-hop smartcomment BFS subgraph around one root node.

    Args:
        graph_path (`str | Path | None`):
            Path to the source smartcomment graph JSON file.
        root_node_id (`str`):
            Root node ID selected by the user.
        bfs_mode (`str`):
            One of `"backward"`, `"forward"`, or `"both"`.

    Returns:
        `RuntimeGraph | None`:
            One-hop BFS runtime subgraph for the selected node, or `None` when
            the root node cannot be resolved.
    """
    loaded = load_exec_network_and_root_variable(graph_path, root_node_id)
    if loaded is None:
        return None
    network, root_node = loaded
    return network.bfs(
        root_node_id,
        direction=bfs_mode,
        max_depth=1,
        include_sibling_inputs=False,
    )


def _build_runtime_nx_graph(subgraph: RuntimeGraph) -> nx.DiGraph:
    """Convert one runtime subgraph into the graph used by Plotly rendering.

    Args:
        subgraph (`RuntimeGraph`):
            Runtime subgraph returned by smartcomment BFS.

    Returns:
        `nx.DiGraph`:
            NetworkX graph with render payloads attached to nodes and edges.
    """
    nx_graph = nx.DiGraph()

    for node in subgraph.nodes:
        node_render = build_runtime_node_render_attrs(node)
        nx_graph.add_node(
            node.full_node_id,
            payload=RelNode(
                node_id=node.full_node_id,
                category=str(node_render["category"]),
                label=str(node_render["label"]),
                color=str(node_render["color"]),
                hover=str(node_render["hover"]),
            ),
            created_at=str(node_render["created_at"]),
        )

    for edge in subgraph.edges:
        edge_render = build_runtime_edge_render_attrs(edge, include_created_at=True)
        if edge.source_full_node_id not in nx_graph or edge.target_full_node_id not in nx_graph:
            continue
        op_id = str(edge_render["op_id"])
        nx_graph.add_edge(
            edge.source_full_node_id,
            edge.target_full_node_id,
            edge_id=edge.edge_id,
            op_id=op_id,
            color=str(edge_render["color"]),
            hover=str(edge_render["hover"]),
        )

    return nx_graph


def render_variable_relation_view(graph: GraphRecord, graph_path: str | Path | None = None) -> None:
    """Render the related-variable overlay for the selected root node.

    Args:
        graph (`GraphRecord`):
            Parsed graph record shown in the current page.
        graph_path (`str | Path | None`, optional):
            Path to the source smartcomment graph JSON file.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    root_node_id = st.session_state.get("current_related_view_root_var")
    if not root_node_id:
        st.info("No variable selected.")
        return

    ensure_related_view_stack(root_node_id)

    selected_key = "related_view_selected_node_id"
    pending_key = "related_view_pending_node_id"
    plot_nonce_key = "related_view_plot_select_nonce"

    mode_key = "current_related_view_bfs_mode"
    current_mode = str(st.session_state.get(mode_key, DEFAULT_RELATED_VIEW_BFS_MODE))
    if current_mode not in VALID_BFS_MODES:
        current_mode = DEFAULT_RELATED_VIEW_BFS_MODE
        st.session_state[mode_key] = current_mode

    header_left, header_back, header_close = st.columns([4, 1, 1])
    with header_left:
        st.subheader(f"{BFS_MODE_LABEL[current_mode]} View")
        st.caption(f"root node: {root_node_id}")
    with header_back:
        if st.button("Return previous node", key="back_related_view_node", use_container_width=True):
            if step_back_related_view():
                st.rerun()

    with header_close:
        if st.button("Close the page", key="close_related_view", use_container_width=True):
            close_related_view()
            st.rerun()

    selected_node_id = str(st.session_state.get(selected_key) or "").strip()
    if selected_node_id:
        st.caption("Selected node_id (click top-right copy button)")
        st.code(selected_node_id, language="text")

    mode_col1, mode_col2, mode_col3 = st.columns(3)
    with mode_col1:
        if st.button(
            "backward",
            key="related_bfs_backward",
            use_container_width=True,
            type="primary" if current_mode == "backward" else "secondary",
        ):
            st.session_state[mode_key] = "backward"
            st.rerun()
    with mode_col2:
        if st.button(
            "forward",
            key="related_bfs_forward",
            use_container_width=True,
            type="primary" if current_mode == "forward" else "secondary",
        ):
            st.session_state[mode_key] = "forward"
            st.rerun()
    with mode_col3:
        if st.button(
            "both",
            key="related_bfs_both",
            use_container_width=True,
            type="primary" if current_mode == "both" else "secondary",
        ):
            st.session_state[mode_key] = "both"
            st.rerun()

    current_mode = str(st.session_state.get(mode_key, current_mode))
    runtime_subgraph = _build_runtime_related_subgraph(graph_path, root_node_id, current_mode)
    if runtime_subgraph is None:
        st.warning("The selected node cannot be resolved in the smartcomment runtime graph.")
        return

    related_edges = runtime_edges_to_payloads(runtime_subgraph)
    with st.expander("All op_id & edge_id in this graph", expanded=False):
        st.markdown(build_op_edge_index_markdown(related_edges))
    if not related_edges:
        st.warning("No dependency edges found under the current BFS mode.")
        return

    nx_graph = _build_runtime_nx_graph(runtime_subgraph)
    if not nx_graph.nodes:
        st.warning("No smartcomment nodes were returned for the selected BFS query.")
        return

    created_at_by_node = {nid: str(data.get("created_at") or "") for nid, data in nx_graph.nodes(data=True)}
    pos = build_ordered_layout(nx_graph, created_at_by_node)

    fig = go.Figure()

    line_segments_by_op: dict[str, dict[str, Any]] = {}
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []
    edge_hit_x: list[float] = []
    edge_hit_y: list[float] = []
    edge_hit_hover: list[str] = []
    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_angles: list[float] = []
    arrow_colors: list[str] = []
    node_points = [(x, y) for x, y in pos.values()]


    for src, dst, data in nx_graph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_geometry = build_edge_render_geometry(x0, y0, x1, y1, node_points)
        op_id = str(data["op_id"])
        edge_color = str(data["color"])
        bucket = line_segments_by_op.setdefault(op_id, {"x": [], "y": [], "color": edge_color})
        bucket["x"].extend([p[0] for p in edge_geometry.path_points] + [None])
        bucket["y"].extend([p[1] for p in edge_geometry.path_points] + [None])

        if edge_geometry.midpoint is not None:
            mid_x, mid_y = edge_geometry.midpoint
            edge_mid_x.append(mid_x)
            edge_mid_y.append(mid_y)
            edge_hover.append(str(data["hover"]))

        end_x, end_y = edge_geometry.arrow_end
        arrow_x.append(end_x)
        arrow_y.append(end_y)
        arrow_angles.append(edge_geometry.arrow_angle)
        arrow_colors.append(edge_color)

        for hx, hy in edge_geometry.hit_points:
            edge_hit_x.append(hx)
            edge_hit_y.append(hy)
            edge_hit_hover.append(str(data["hover"]))

    for op_id in sorted(line_segments_by_op):
        segment = line_segments_by_op[op_id]
        fig.add_trace(
            go.Scattergl(
                x=segment["x"],
                y=segment["y"],
                mode="lines",
                line=dict(color=segment["color"], width=2),
                hoverinfo="skip",
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

    fig.add_trace(
        go.Scattergl(
            x=edge_mid_x,
            y=edge_mid_y,
            mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0.001)"),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=edge_hover,
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=edge_hit_x,
            y=edge_hit_y,
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0.001)"),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=edge_hit_hover,
            showlegend=False,
        )
    )

    category_to_color: dict[str, str] = {}
    node_hit_x: list[float] = []
    node_hit_y: list[float] = []
    node_hit_hover: list[str] = []
    node_hit_custom: list[str] = []
    nodes_by_category: dict[str, list[tuple[float, float, str, str, str]]] = {}

    for node_id, data in nx_graph.nodes(data=True):
        payload: RelNode = data["payload"]
        x, y = pos[node_id]
        category_to_color[payload.category] = payload.color
        hover_text = payload.hover or f"<b>node_id</b>: {node_id}"
        node_custom = f"node|{node_id}"
        nodes_by_category.setdefault(payload.category, []).append(
            (x, y, payload.color, hover_text, node_custom)
        )
        node_hit_x.append(x)
        node_hit_y.append(y)
        node_hit_hover.append(hover_text)
        node_hit_custom.append(node_custom)

    for category in sorted(nodes_by_category):
        items = nodes_by_category[category]
        fig.add_trace(
            go.Scattergl(
                x=[it[0] for it in items],
                y=[it[1] for it in items],
                mode="markers",
                marker=dict(size=24, color=[it[2] for it in items], line=dict(color="#111827", width=1.2)),
                customdata=[it[4] for it in items],
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=[it[3] for it in items],
                name=category,
                legendgroup=f"cat::{category}",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scattergl(
            x=node_hit_x,
            y=node_hit_y,
            mode="markers",
            marker=dict(size=40, color="rgba(0,0,0,0.001)"),
            customdata=node_hit_custom,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=node_hit_hover,
            showlegend=False,
        )
    )

    fig.update_layout(
        height=760,
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
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        clickmode="event+select",
        dragmode="pan",
        hovermode="closest",
        hoverlabel=dict(
            font_size=13,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        uirevision=f"related_{graph.graph_id}_{root_node_id}",
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"related_view_plot_{graph.graph_id}_{root_node_id}_{int(st.session_state.get(plot_nonce_key, 0))}",
        on_select="rerun",
        selection_mode=["points"],
    )

    points = extract_selected_points(event)
    if not points:
        return

    clicked_node_id = pick_clicked_node_id(points)
    if not clicked_node_id:
        return

    st.session_state[selected_key] = clicked_node_id
    pending_node_id = str(st.session_state.get(pending_key) or "").strip()
    if clicked_node_id == root_node_id:
        if pending_node_id != clicked_node_id:
            st.session_state[pending_key] = clicked_node_id
            # Force a fresh plot key so clicking the root node can reveal its copyable ID.
            st.session_state[plot_nonce_key] = int(st.session_state.get(plot_nonce_key, 0)) + 1
            st.rerun()
        return

    if pending_node_id != clicked_node_id:
        st.session_state[pending_key] = clicked_node_id
        # Force a fresh plot key so clicking the same node again can trigger a new select event.
        st.session_state[plot_nonce_key] = int(st.session_state.get(plot_nonce_key, 0)) + 1
        st.rerun()

    push_related_view_node(clicked_node_id)
    st.rerun()
