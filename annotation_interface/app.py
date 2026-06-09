"""Streamlit entry for LLM memory graph visualization."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import streamlit as st
import argparse

from components.op_detail_view import render_op_detail_view
from components.error_attribution_view import render_error_attribution_view
from components.sidebar_widgets import render_sidebar_widgets
from components.timeline_plotter import render_timeline_plot
from components.variable_relation_view import render_variable_relation_view
from config import COLOR_ASSISTANT, COLOR_SYSTEM, COLOR_USER
from data_engine.dataset_index import (
    build_data_version_token,
    build_dataset_index_file,
    dataset_index_path,
    is_dataset_index_stale,
    load_dataset_index_file,
    load_graph_by_name,
)
from utils.session_state import init_session_state, reset_view_state_for_graph_change


@st.cache_data(show_spinner=False)
def _load_graphs_cached(dataset_path: str, data_version: str) -> list[dict[str, Any]] | None:
    """Load or rebuild the dataset index used by the top-level graph selector.

    Args:
        dataset_path (`str`):
            Path to a graph JSON file or a directory that contains graph JSON files.
        data_version (`str`):
            Cache-busting token derived from source file metadata.

    Returns:
        `list[dict[str, Any]] | None`:
            Indexed graph metadata used by the Streamlit UI, or `None` when the
            index file cannot be loaded.
    """
    _ = data_version  # cache-busting token derived from file mtime
    path = Path(dataset_path)
    index_path = dataset_index_path(path)

    if is_dataset_index_stale(path, index_path):
        with st.spinner("Building dataset index automatically... This may take a while for large datasets. Please check the terminal for progress."):
            build_dataset_index_file(dataset_path, index_path)

    return load_dataset_index_file(index_path)

@st.cache_data(show_spinner=False)
def _load_single_graph_cached(graph_path: str, target_graph_name: str, data_version: str) -> Any:
    """Load one parsed graph record from a source JSON file.

    Args:
        graph_path (`str`):
            Path to the source JSON file.
        target_graph_name (`str`):
            Display name shown in the selector, including optional `#<index>`
            suffix for multi-graph files.
        data_version (`str`):
            Cache-busting token derived from source file metadata.

    Returns:
        `Any`:
            The parsed `GraphRecord` matching `target_graph_name`, or `None`
            when no matching graph exists in the file.
    """
    # Actually data_version is global, which lets this update if dir changes.
    _ = data_version
    return load_graph_by_name(graph_path, target_graph_name)


def _render_legend() -> None:
    """Render the global node-type legend displayed under the page hero.

    Args:
        None.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    st.markdown("<div class='legend-title'>Variables</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='legend-row'>"
        f"<span class='var-chip'><span style='color:{COLOR_USER}'>●</span> User</span>"
        f"<span class='var-chip'><span style='color:{COLOR_ASSISTANT}'>●</span> Assistant</span>"
        f"<span class='var-chip'><span style='color:{COLOR_SYSTEM}'>●</span> System</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _inject_theme_styles() -> None:
    """Inject the global Streamlit theme overrides used by the app.

    Args:
        None.

    Returns:
        `None`:
            This function injects CSS into the current Streamlit page.
    """
    st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
    --bg: #f4f6f8;
    --surface: #ffffff;
    --surface-soft: #f8fafb;
    --text: #1f2937;
    --muted: #64748b;
    --accent: #2563eb;
    --stroke: #dbe3ea;
    --shadow-rgb: 32, 66, 104;
}

html, body, [class*="css"] {
    font-family: 'Outfit', 'Noto Sans SC', sans-serif;
    color: var(--text);
    scroll-behavior: smooth;
}

.stApp {
    position: relative;
    background:
        radial-gradient(1200px 420px at 0% -10%, rgba(37,99,235,0.07), transparent 60%),
        radial-gradient(1000px 380px at 100% -20%, rgba(16,185,129,0.06), transparent 58%),
        var(--bg);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(rgba(255,255,255,0.06) 0.6px, transparent 0.6px),
        radial-gradient(rgba(0,0,0,0.05) 0.5px, transparent 0.5px);
    background-position: 0 0, 14px 11px;
    background-size: 24px 24px, 28px 28px;
    opacity: 0.22;
    z-index: 0;
}

.stApp > header,
.stApp > [data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
}

/* Remove Streamlit top action bar entries such as Deploy and overflow menu. */
header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

#MainMenu {
    display: none !important;
}

.block-container {
    max-width: 95%;
    padding-top: 1.3rem;
    padding-bottom: 1.8rem;
}

.hero-shell {
    border: 1px solid var(--stroke);
    background: linear-gradient(120deg, color-mix(in srgb, var(--surface) 92%, #ffffff 8%), color-mix(in srgb, var(--surface-soft) 95%, #ffffff 5%));
    border-radius: 16px;
    padding: 0.85rem 1rem 0.95rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.16);
    margin-bottom: 0.72rem;
}

.hero-title {
    margin: 0;
    font-size: 2.50rem;
    line-height: 1.2;
}

.hero-sub {
    margin-top: 0.32rem;
    color: var(--muted);
    font-size: 0.95rem;
    max-width: 100ch;
    line-height: 1.5;
}

.legend-title {
    margin-top: 0.68rem;
    margin-bottom: 0.3rem;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    color: var(--muted);
    font-weight: 700;
}

.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
    text-wrap: balance;
}

h1 {
    font-weight: 700;
}

.var-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.58rem;
    margin-right: 0.38rem;
    border: 1px solid var(--stroke);
    border-radius: 999px;
    background: color-mix(in srgb, var(--surface) 90%, transparent);
    font-size: 0.84rem;
    color: var(--text);
    backdrop-filter: blur(4px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    animation: panel-enter 260ms ease-out;
}

@keyframes panel-enter {
    from {
        opacity: 0;
        transform: translateY(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.qa-card {
    border: 1px solid var(--stroke);
    background: var(--surface);
    border-radius: 11px;
    padding: 0.56rem 0.62rem 0.5rem;
    margin-bottom: 0.28rem;
    box-shadow: 0 8px 22px rgba(var(--shadow-rgb), 0.08);
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}

.qa-card:hover {
    transform: translateY(-1px);
    border-color: color-mix(in srgb, var(--accent) 30%, var(--stroke) 70%);
    box-shadow: 0 12px 28px rgba(var(--shadow-rgb), 0.13);
}

.qa-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}

.qa-no {
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 600;
}

.qa-status {
    font-size: 0.76rem;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    font-weight: 600;
}

.qa-status.ok {
    background: rgba(22, 163, 74, 0.12);
    color: #15803d;
}

.qa-status.bad {
    background: rgba(220, 38, 38, 0.12);
    color: #b91c1c;
}

.qa-q {
    font-size: 0.88rem;
    color: var(--text);
    margin-bottom: 0.2rem;
}

.qa-label {
    font-size: 0.74rem;
    letter-spacing: 0.03em;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 0.16rem;
}

.qa-a {
    font-size: 0.84rem;
    color: color-mix(in srgb, var(--text) 80%, var(--muted) 20%);
    line-height: 1.38;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
    border-color: var(--stroke);
    background: var(--surface);
    box-shadow:
        0 12px 30px rgba(var(--shadow-rgb), 0.12),
        inset 0 1px 0 rgba(255,255,255,0.18);
}

[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--stroke) 84%, transparent);
}

label[data-testid="stWidgetLabel"] {
    font-weight: 600;
    color: color-mix(in srgb, var(--text) 84%, var(--muted) 16%);
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: color-mix(in srgb, var(--stroke) 88%, #c8d6e8 12%);
    background: var(--surface);
}

[data-testid="stRadio"] [role="radiogroup"] {
    background: color-mix(in srgb, var(--surface) 74%, transparent);
    border: 1px solid var(--stroke);
    border-radius: 10px;
    padding: 0.16rem 0.26rem;
    display: flex;
    flex-wrap: nowrap;
    gap: 0.28rem;
    overflow-x: auto;
    scrollbar-width: thin;
}

[data-testid="stRadio"] label {
    margin-bottom: 0;
    white-space: nowrap;
}

button[kind="secondary"] {
    border-radius: 10px;
    border: 1px solid color-mix(in srgb, var(--stroke) 80%, var(--accent) 20%);
    background: color-mix(in srgb, var(--surface) 92%, #ffffff 8%);
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

button[kind="secondary"]:hover {
    border-color: color-mix(in srgb, var(--accent) 54%, var(--stroke) 46%);
    box-shadow: 0 10px 22px rgba(var(--shadow-rgb), 0.14);
    transform: translateY(-1px);
}

button[kind="secondary"]:active {
    transform: translateY(0);
}

button:focus-visible, [role="radiogroup"]:focus-visible {
    outline: 2px solid rgba(37,99,235,0.35) !important;
    outline-offset: 2px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--stroke);
    min-width: 460px;
    max-width: 460px;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 8px;
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}

.qa-scroll-caption {
    margin-top: -0.2rem;
    margin-bottom: 0.35rem;
    color: var(--muted);
    font-size: 0.78rem;
}

[data-testid="stVerticalBlock"] .qa-scroll-host [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.52;
    max-width: 70ch;
}

[data-testid="stCaptionContainer"] {
    font-variant-numeric: tabular-nums;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def parse_args() -> argparse.Namespace:
    """Parse Streamlit passthrough CLI arguments for the web app.

    Args:
        None.

    Returns:
        `argparse.Namespace`:
            Parsed CLI arguments for dataset path, API config path, and output path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", 
        type=str, 
        default="/dataset",
        help="Path to a graph JSON file or to a directory containing multiple graph JSON files"
    )
    parser.add_argument(
        "--api-config", 
        type=str, 
        default="api_config.json",
        help="Path to the api config file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="error_annotation.json",
        help="Path to save error attribution case records as a single JSON array"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Judge model name passed to the error attribution runner",
    )
    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    """Run the Streamlit memory visualization app entrypoint.

    Args:
        None.

    Returns:
        `None`:
            This function boots the Streamlit UI and renders the selected graph.
    """
    args = parse_args()
    data_path = Path(args.data)
    api_config_path = Path(args.api_config)
    output_path = Path(args.output_path)
    judge_model = str(args.judge_model)

    st.set_page_config(page_title="LLM Memory Visualizer", layout="wide")

    init_session_state()
    _inject_theme_styles()    

    if not data_path.exists():
        st.error(f"Data path does not exist: {data_path}")
        return

    data_version = build_data_version_token(data_path)
    graph_entries = _load_graphs_cached(str(data_path), data_version)
    
    if graph_entries is None:
        st.error("Failed to load or build the dataset index from the target directory.")
        st.stop()

    st.markdown(
        "<div class='hero-shell'>"
        "<h1 class='hero-title'>LLM Agent Memory Flow Visualizer</h1>"
        "<div class='hero-sub'>Explore session-level graphs, memory interactions, and QA evidence links in one synchronized dashboard.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_legend()

    selected_graph = None
    selected_graph_path: Path | None = None
    with st.container(border=True):
        st.markdown("**Controls**")
        if graph_entries:
            graph_names = [entry["graph_name"] for entry in graph_entries]
            pending_jump_graph_id = st.session_state.get("pending_jump_graph_id")
            if pending_jump_graph_id in graph_names and st.session_state.get("top_graph_select") != pending_jump_graph_id:
                st.session_state["top_graph_select"] = pending_jump_graph_id

            current_id = st.session_state.get("current_graph_id")
            if current_id not in graph_names:
                current_id = pending_jump_graph_id if pending_jump_graph_id in graph_names else graph_names[0]
            if st.session_state.get("top_graph_select") not in graph_names:
                st.session_state["top_graph_select"] = current_id

            wrong_qa_options: list[tuple[str, str]] = []
            wrong_qa_lookup: dict[str, dict[str, str]] = {}
            for entry in graph_entries:
                graph_name = entry["graph_name"]
                for qa_info in entry.get("wrong_qas", []):
                    qa_name = qa_info["query_full_name"]
                    qa_id = qa_info["qa_id"]
                    option_value = f"{graph_name}@@{qa_id}"
                    option_label = f"{qa_name}"
                    wrong_qa_options.append((option_value, option_label))
                    wrong_qa_lookup[option_value] = {
                        "graph_name": graph_name,
                        "qa_id": qa_id,
                    }

            wrong_qa_options.sort(key=lambda x: x[1])
            rng = random.Random(42)
            rng.shuffle(wrong_qa_options)

            wrong_qa_values = [""] + [value for value, _ in wrong_qa_options]
            wrong_qa_label_map = {value: label for value, label in wrong_qa_options}

            graph_col, query_col = st.columns([1.5, 1.5], gap="medium")
            with graph_col:
                selected_id = st.selectbox(
                    "Graph",
                    graph_names,
                    key="top_graph_select",
                )
            reset_view_state_for_graph_change(selected_id)

            if pending_jump_graph_id and selected_id == pending_jump_graph_id:
                pending_jump_qa_id = st.session_state.get("pending_jump_qa_id")
                if pending_jump_qa_id:
                    st.session_state["qa_filter"] = "wrong"
                    st.session_state["pending_sidebar_qa_id"] = pending_jump_qa_id
                st.session_state["pending_jump_graph_id"] = None
                st.session_state["pending_jump_qa_id"] = None

            selected_entry = next(entry for entry in graph_entries if entry["graph_name"] == selected_id)
            selected_graph = _load_single_graph_cached(
                str(selected_entry["graph_path"]),
                str(selected_entry["graph_name"]),
                data_version,
            )
            selected_graph_path = Path(str(selected_entry["graph_path"]))

            with query_col:
                selected_wrong_qa = st.selectbox(
                    "Query",
                    options=wrong_qa_values,
                    key="top_wrong_qa_select",
                    format_func=lambda value: wrong_qa_label_map.get(value, "(Select a wrong QA)") if value else "(Select a wrong QA)",
                )

            if not selected_wrong_qa:
                st.session_state["last_applied_top_wrong_qa"] = None

            if selected_wrong_qa and st.session_state.get("last_applied_top_wrong_qa") != selected_wrong_qa:
                target = wrong_qa_lookup.get(selected_wrong_qa)
                if target is not None:
                    target_graph_name = target["graph_name"]
                    target_qa_id = target["qa_id"]
                    st.session_state["pending_jump_graph_id"] = target_graph_name
                    st.session_state["pending_jump_qa_id"] = target_qa_id
                    st.session_state["related_view_active"] = False
                    st.session_state["attribution_view_active"] = False
                    st.session_state["last_applied_top_wrong_qa"] = selected_wrong_qa
                    st.rerun()
        else:
            st.info("No graph data loaded.")

    # different views
    related_active = st.session_state.get("related_view_active")
    attribution_active = st.session_state.get("attribution_view_active")
    expanded_right_overlay = bool(related_active or attribution_active)

    if expanded_right_overlay:
        col_left, col_main = st.columns([1.25, 4.75], gap="small")
        col_right = None
    else:
        col_left, col_main, col_right = st.columns([1.25, 2.7, 2.05], gap="small")

    with col_left:
        render_sidebar_widgets(selected_graph)

    if selected_graph is None:
        st.info("Currently, there is no traceable graph data available.")
        return

    if attribution_active:
        with col_main:
            render_error_attribution_view(
                data_path=selected_graph_path,
                api_config_path=api_config_path,
                output_path=output_path,
                graph_id=selected_graph.graph_id,
                judge_model=judge_model,
                graph=selected_graph,
            )
    elif related_active:
        with col_main:
            render_variable_relation_view(
                graph=selected_graph,
                graph_path=selected_graph_path
            )
    else:
        with col_main:
            render_timeline_plot(graph=selected_graph)

        with col_right:
            render_op_detail_view(
                graph=selected_graph,
                graph_path=selected_graph_path
            )


if __name__ == "__main__":
    main()
