"""Error attribution overlay view for wrong QA items."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import streamlit as st
import plotly.graph_objects as go
from error_attribution.trace_error_attribution import (
    TraceErrorAttributionRunner,
    TraceErrorAttributionRunnerConfig,
    build_construction_subgraph,
    build_response_subgraph,
    build_retrieval_subgraph,
    collect_failed_cases,
    resolve_latest_nodes_by_full_name,
)

from utils.graph_render import (
    build_edge_render_geometry,
    build_ordered_layout,
    build_runtime_edge_hover,
    build_runtime_node_hover,
    edge_color_from_op_id,
    extract_selected_points,
    node_color_from_category,
    pick_clicked_node_id,
)
from utils.runtime_graph import build_op_edge_index_markdown, load_all_graph_op_edge_ids, runtime_graph_to_payload
from utils.session_state import activate_related_view, close_attribution_view
from data_engine.parser import GraphRecord


MANUAL_ERROR_TYPE_OPTIONS = [
    "AnnotationError",
    "LLMAsAJudgeError",
    "ExtractionError",
    "UpdateError",
    "DeletionError",
    "RetrievalError",
    "ResponseError",
]


def _find_failed_case(
    failed_cases: list[Any],
    *,
    query_full_name: str | None,
    query_node_id: str | None,
) -> Any | None:
    """Find the failed case matching the selected query identifier."""
    for case_obj in failed_cases:
        if query_node_id and case_obj.query_node.full_node_id == query_node_id:
            return case_obj
        if query_full_name and case_obj.query_node.full_name == query_full_name:
            return case_obj
    return None


def _attach_case_graph_payloads(
    runner: TraceErrorAttributionRunner,
    network: Any,
    target_case_obj: Any,
    target_case_dict: dict[str, Any],
) -> None:
    """Populate retrieval/response/construction graph payloads in one case dict."""
    retrieval_subgraph = build_retrieval_subgraph(network, target_case_obj)
    response_subgraph = build_response_subgraph(network, target_case_obj)

    target_case_dict["retrieval_input"]["retrieval_graph_payload"] = runtime_graph_to_payload(retrieval_subgraph)
    target_case_dict["response_input"]["response_graph_payload"] = runtime_graph_to_payload(response_subgraph)

    source_evidence_full_names = target_case_dict.get("source_evidence_full_names", [])
    source_evidence_nodes = resolve_latest_nodes_by_full_name(network, source_evidence_full_names)
    name_to_node_map = {node.full_name: node for node in source_evidence_nodes}

    for cons_input in target_case_dict["construction_inputs"]:
        evidence_name = cons_input["source_evidence_full_name"]
        node_obj = name_to_node_map.get(evidence_name)
        if node_obj is None:
            continue
        session_graph = runner._get_session_graph(node_obj.session_id)
        cons_sg = build_construction_subgraph(network, session_graph, node_obj)
        cons_input["construction_graph_payload"] = runtime_graph_to_payload(cons_sg)


@st.cache_data(show_spinner=False)
def _run_trace_attribution(
    graph_path: str,
    judge_model: str,
    api_config_path: str | None,
    query_full_name: str | None,
    query_node_id: str | None,
    rerun_nonce: int = 0,
) -> dict[str, Any] | None:
    """Run trace-based attribution for one selected failed QA case.

    Args:
        graph_path (`str`):
            Path to the source smartcomment graph JSON file.
        judge_model (`str`):
            LLM model name used for attribution judging.
        api_config_path (`str | None`):
            Optional API config file path for the LLM backend.
        query_full_name (`str | None`):
            Full query variable name used to identify the target failed case.
        query_node_id (`str | None`):
            Full query node ID used to identify the target failed case.
        rerun_nonce (`int`, defaults to `0`):
            Cache-busting nonce used when the user explicitly requests a rerun.

    Returns:
        `dict[str, Any] | None`:
            One attribution result payload augmented with graph payloads, or
            `None` when the target failed case is not found.
    """
    _ = rerun_nonce
    config = TraceErrorAttributionRunnerConfig(
        graph_path=graph_path,
        judge_model=judge_model,
        api_config_path=api_config_path,
        include_metadata_in_graph_text=True,
        output_path=None,
    )
    runner = TraceErrorAttributionRunner(config)

    network = runner._load_graph()
    failed_cases = collect_failed_cases(network)
    target_case_obj = _find_failed_case(
        failed_cases,
        query_full_name=query_full_name,
        query_node_id=query_node_id,
    )
    if target_case_obj is None:
        return None

    target_case_dict = runner.prepare_single_case(network, target_case_obj)
    _attach_case_graph_payloads(runner, network, target_case_obj, target_case_dict)
    result = runner.run_single_case(target_case_dict)
    return result


def _render_candidate_ops_with_edges(
    *,
    title: str,
    candidate_op_ids: Any,
    graph_payload: dict[str, Any] | None,
) -> None:
    """Render a collapsible `op_id -> edge_id` index for one attribution subgraph.

    Args:
        title (`str`):
            Expander title shown above the grouped IDs.
        candidate_op_ids (`Any`):
            Candidate operation IDs returned by the attribution runner.
        graph_payload (`dict[str, Any] | None`):
            Serialized subgraph payload used to recover edge IDs.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    with st.expander(title, expanded=False):
        edge_payloads = graph_payload.get("edges", []) if isinstance(graph_payload, dict) else []
        st.markdown(build_op_edge_index_markdown(edge_payloads, candidate_op_ids))


def _load_existing_case_results(output_path: Path) -> list[dict[str, Any]]:
    """Load previously saved case records from one JSON array file.

    Args:
        output_path (`Path`):
            Output JSON file that stores persisted attribution case records.

    Returns:
        `list[dict[str, Any]]`:
            Normalized case records loaded from the output file.
    """
    if not output_path.exists():
        return []

    raw_text = output_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    loaded = json.loads(raw_text)

    if isinstance(loaded, list):
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(loaded, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Invalid JSON array item at index {idx}: each case must be a JSON object."
                )
            normalized.append(item)
        return normalized

    raise RuntimeError("Output file must contain a JSON array of case objects.")


def _append_case_result(output_path: Path, case_payload: dict[str, Any]) -> None:
    """Append one case payload into a single JSON array file.

    Args:
        output_path (`Path`):
            Output JSON file that stores persisted attribution case records.
        case_payload (`dict[str, Any]`):
            Case payload that should be appended to the saved JSON array.

    Returns:
        `None`:
            This function rewrites the output file with the appended case record.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_results = _load_existing_case_results(output_path)
    case_results.append(case_payload)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(case_results, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _render_graph_payload(title: str, graph_payload: dict[str, Any], key: str) -> None:
    """Render one attribution subgraph payload as an interactive Plotly graph.

    Args:
        title (`str`):
            Graph title shown above the Plotly figure.
        graph_payload (`dict[str, Any]`):
            Serialized node and edge payload returned by `_runtime_graph_to_payload`.
        key (`str`):
            Stable Streamlit widget key suffix for this subgraph instance.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    if not nodes:
        st.caption(f"{title}: no nodes")
        return

    graph = nx.DiGraph()
    for n in nodes:
        node_id = n["id"]
        category = str(n.get("category") or n.get("class_name") or "").strip() or "unknown"
        color = node_color_from_category(category)

        graph.add_node(
            node_id,
            created_at=str(n.get("created_at") or ""),
            category=category,
            color=color,
            hover=build_runtime_node_hover(
                full_node_id=node_id,
                full_name=str(n.get("full_name", "") or ""),
                category=category,
                class_name=str(n.get("class_name", "") or ""),
                comment=str(n.get("comment", "") or ""),
                value=n.get("value"),
            ),
        )

    for e in edges:
        src = e.get("source")
        dst = e.get("target")
        if src not in graph or dst not in graph:
            continue
        op_id = str(e.get("op_id") or "")
        graph.add_edge(
            src,
            dst,
            op_id=op_id,
            color=edge_color_from_op_id(op_id),
            hover=build_runtime_edge_hover(
                edge_id=str(e.get("edge_id", "") or ""),
                op_id=op_id,
                category=str(e.get("category", "") or ""),
                source_id=str(e.get("source", "") or ""),
                target_id=str(e.get("target", "") or ""),
                comment=str(e.get("comment", "") or ""),
                source_label="source",
                target_label="target",
            ),
        )

    created_at_by_node = {nid: str(data.get("created_at") or "") for nid, data in graph.nodes(data=True)}
    pos = build_ordered_layout(graph, created_at_by_node)

    edge_segments_by_op: dict[str, dict[str, Any]] = {}
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []
    edge_hit_x: list[float] = []
    edge_hit_y: list[float] = []
    edge_hit_hover: list[str] = []

    node_points = [(x, y) for x, y in pos.values()]

    for src, dst, data in graph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_geometry = build_edge_render_geometry(x0, y0, x1, y1, node_points)
        op_id = str(data.get("op_id") or "")
        bucket = edge_segments_by_op.setdefault(
            op_id,
            {"x": [], "y": [], "color": str(data.get("color") or edge_color_from_op_id(op_id))},
        )
        bucket["x"].extend([p[0] for p in edge_geometry.path_points] + [None])
        bucket["y"].extend([p[1] for p in edge_geometry.path_points] + [None])

        if edge_geometry.midpoint is not None:
            mid_x, mid_y = edge_geometry.midpoint
            edge_mid_x.append(mid_x)
            edge_mid_y.append(mid_y)
            edge_hover.append(data.get("hover", ""))

        hover_text = data.get("hover", "")
        for hx, hy in edge_geometry.hit_points:
            edge_hit_x.append(hx)
            edge_hit_y.append(hy)
            edge_hit_hover.append(hover_text)

    node_hit_x: list[float] = []
    node_hit_y: list[float] = []
    node_hit_hover: list[str] = []
    node_hit_custom: list[str] = []
    nodes_by_category: dict[str, list[tuple[float, float, str, str, str]]] = {}

    for nid, data in graph.nodes(data=True):
        x, y = pos[nid]
        category = str(data.get("category") or "").strip() or "unknown"
        color = str(data.get("color", "#6b7280"))
        hover_text = data.get("hover") or f"<b>node_id</b>: {nid}"
        node_custom = f"node|{nid}"
        nodes_by_category.setdefault(category, []).append((x, y, color, hover_text, node_custom))
        node_hit_x.append(x)
        node_hit_y.append(y)
        node_hit_hover.append(hover_text)
        node_hit_custom.append(node_custom)

    fig = go.Figure()
    for op_id in sorted(edge_segments_by_op):
        segment = edge_segments_by_op[op_id]
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
    plot_nonce_key = "attribution_plot_select_nonce"
    selected_key = "attribution_selected_node_id"
    pending_key = "attribution_pending_node_id"
    selected_source_key = "attribution_selected_source_key"
    pending_source_key = "attribution_pending_source_key"

    selected_node_id = str(st.session_state.get(selected_key) or "").strip()
    selected_source_key_value = str(st.session_state.get(selected_source_key) or "")
    if selected_node_id and selected_source_key_value == key:
        st.caption("Selected node_id (click top-right copy button)")
        st.code(selected_node_id, language="text")

    fig.update_layout(
        title=title,
        height=760,
        showlegend=bool(nodes_by_category),
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
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision=f"attribution_{key}",
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_{int(st.session_state.get(plot_nonce_key, 0))}",
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
    st.session_state[selected_source_key] = key
    pending_node_id = str(st.session_state.get(pending_key) or "").strip()
    pending_source_key_value = str(st.session_state.get(pending_source_key) or "")
    if pending_node_id != clicked_node_id or pending_source_key_value != key:
        st.session_state[pending_key] = clicked_node_id
        st.session_state[pending_source_key] = key
        # Force a fresh plot key so second click on the same node creates a new select event.
        st.session_state[plot_nonce_key] = int(st.session_state.get(plot_nonce_key, 0)) + 1
        st.rerun()

    activate_related_view(clicked_node_id, return_target="attribution")
    st.session_state[pending_key] = None
    st.session_state[pending_source_key] = None
    st.session_state["attribution_view_active"] = False
    st.rerun()


def render_error_attribution_view(
    *,
    data_path: Path,
    api_config_path: Path,
    output_path: Path | None,
    graph_id: str,
    judge_model: str,
    graph: GraphRecord | None = None,
) -> None:
    """Render the error-attribution overlay for one selected failed QA case.

    Args:
        data_path (`Path`):
            Path to the source graph JSON file.
        api_config_path (`Path`):
            Path to the API config file used by the attribution runner.
        output_path (`Path | None`):
            Optional JSON output file used to store manual annotations.
        graph_id (`str`):
            Display graph identifier shown in the overlay header.
        judge_model (`str`):
            LLM judge model name forwarded to the attribution runner.
        graph (`GraphRecord | None`, optional):
            Parsed graph record for the current page, used for validation and UI details.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    qa_id = st.session_state.get("current_attribution_qa_id")
    query_full_name = st.session_state.get("current_attribution_query_full_name")
    query_node_id = st.session_state.get("current_attribution_query_node_id")
    rerun_nonce = int(st.session_state.get("current_attribution_rerun_nonce", 0))

    header_left, header_rerun, header_close = st.columns([4, 1, 1])
    with header_left:
        st.subheader("Error Attribution View")
        st.caption(
            f"Graph: {graph_id} | QA: {qa_id or '(unknown)'} | Query: {query_full_name or query_node_id or '(unknown)'}"
        )
    with header_rerun:
        if st.button("Re-run attribution", key="rerun_error_attribution_view", use_container_width=True):
            st.session_state["current_attribution_rerun_nonce"] = rerun_nonce + 1
            st.rerun()
    with header_close:
        if st.button("Close the page", key="close_error_attribution_view", use_container_width=True):
            close_attribution_view()
            st.rerun()

    if output_path is not None:
        st.caption(f"Output file (JSON array): {output_path}")
    else:
        st.caption("Output file is not set. Start app with --output-path to enable file export.")

    if not data_path.exists():
        st.error(f"Graph file not found: {data_path}")
        return

    with st.spinner("Running trace error attribution..."):
        try:
            selected_result = _run_trace_attribution(
                graph_path=str(data_path),
                judge_model=judge_model,
                api_config_path=str(api_config_path),
                query_full_name=query_full_name,
                query_node_id=query_node_id,
                rerun_nonce=rerun_nonce,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error attribution failed: {exc}")
            return

    if selected_result is None:
        st.warning("No failed-case attribution input found for the selected QA query.")
        return

    llm_inputs = selected_result.get("llm_inputs", {})
    all_graph_op_ids, all_graph_edge_ids = load_all_graph_op_edge_ids(str(data_path))
    display_result = {
        "query": selected_result.get("query"),
        "golden_answer": selected_result.get("golden_answer"),
        "prediction": selected_result.get("prediction"),
        "error_type": selected_result.get("error_type"),
        "error_details": selected_result.get("error_details"),
    }

    st.markdown("### 1) Manual Error Annotation")
    input_key_suffix = query_node_id or qa_id or "na"
    input_col_left, input_col_right = st.columns(2)
    with input_col_left:
        manual_op_id = st.text_input(
            "op_id",
            key=f"attr_manual_op_id_{input_key_suffix}",
            placeholder="e.g. op-xxx or null",
            help=f"Must be 'null' or a valid op_id from the current graph.",
        )
        manual_edge_id = st.text_input("edge_id", key=f"attr_manual_edge_id_{input_key_suffix}")
    with input_col_right:
        manual_error_type = st.selectbox(
            "error_type",
            options=[""] + MANUAL_ERROR_TYPE_OPTIONS,
            key=f"attr_manual_error_type_{input_key_suffix}",
            format_func=lambda value: value if value else "(Select an error_type)",
        )
        manual_reason = st.text_area("reason", key=f"attr_manual_reason_{input_key_suffix}", height=96)

    if st.button("Append current case to output file", key=f"append_attr_case_{input_key_suffix}"):
        if output_path is None:
            st.error("output_path is not configured. Please restart with --output-path.")
        else:
            normalized_manual_op_id = manual_op_id.strip() if manual_op_id else ""
            if normalized_manual_op_id and normalized_manual_op_id != "null" and normalized_manual_op_id not in all_graph_op_ids:
                st.error("Invalid op_id: it does not exist in the current graph.")

            elif manual_edge_id and manual_edge_id.strip() not in all_graph_edge_ids:
                st.error("Invalid edge_id: it does not exist in the current graph.")
            else:
                case_payload = {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "graph_id": graph_id,
                    "query_id": query_node_id or qa_id,
                    "query": display_result.get("query"),
                    "manual_error_attribution": {
                        "op_id": manual_op_id,
                        "error_type": manual_error_type,
                        "edge_id": manual_edge_id,
                        "reason": manual_reason,
                    },
                }
                try:
                    _append_case_result(output_path, case_payload)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to append case to output file: {exc}")
                else:
                    st.success(f"Saved current case to: {output_path}")

    st.markdown("### 2) Error Attribution JSON Output")
    st.json(display_result)

    st.markdown("### 3) Construction Subgraphs By Evidence")
    construction_inputs = llm_inputs.get("construction_inputs", [])
    if not construction_inputs:
        st.caption("No construction subgraph available for this QA.")
    else:
        for idx, construction in enumerate(construction_inputs, start=1):
            source_name = construction.get("source_evidence_full_name", "(unknown)")
            with st.expander(f"Evidence {idx}: {source_name}", expanded=False):
                _render_candidate_ops_with_edges(
                    title="Candidate op_id & edge_id",
                    candidate_op_ids=construction.get("candidate_op_ids", ""),
                    graph_payload=construction.get("construction_graph_payload", {}),
                )
                _render_graph_payload(
                    title=f"Construction Subgraph - Evidence {idx}",
                    graph_payload=construction.get("construction_graph_payload", {}),
                    key=f"attr_cons_graph_{idx}_{qa_id or 'na'}",
                )

    st.markdown("### 4) Retrieval Subgraph")
    retrieval_input = llm_inputs.get("retrieval_input", {})
    _render_candidate_ops_with_edges(
        title="Candidate op_id & edge_id",
        candidate_op_ids=retrieval_input.get("candidate_op_ids", ""),
        graph_payload=retrieval_input.get("retrieval_graph_payload", {}),
    )
    _render_graph_payload(
        title="Retrieval Subgraph",
        graph_payload=retrieval_input.get("retrieval_graph_payload", {}),
        key=f"attr_retrieval_graph_{qa_id or 'na'}",
    )

    st.markdown("### 5) Response Subgraph")
    response_input = llm_inputs.get("response_input", {})
    _render_candidate_ops_with_edges(
        title="Candidate op_id & edge_id",
        candidate_op_ids=response_input.get("candidate_op_ids", ""),
        graph_payload=response_input.get("response_graph_payload", {}),
    )
    _render_graph_payload(
        title="Response Subgraph",
        graph_payload=response_input.get("response_graph_payload", {}),
        key=f"attr_response_graph_{qa_id or 'na'}",
    )
