"""Session state utilities for Streamlit interaction state."""

from __future__ import annotations

from typing import Any

import streamlit as st

# Allowed QA filter values used by sidebar widgets.
QA_FILTER_ALL = "all"
QA_FILTER_CORRECT = "correct"
QA_FILTER_WRONG = "wrong"

DEFAULT_QA_FILTER = QA_FILTER_ALL
DEFAULT_RELATED_VIEW_BFS_MODE = "backward"


def _related_view_defaults() -> dict[str, Any]:
    """Return default session-state values for the related-variable view."""
    return {
        "related_view_active": False,
        "current_related_view_root_var": None,
        "related_view_node_stack": [],
        "related_view_selected_node_id": None,
        "related_view_pending_node_id": None,
        "related_view_plot_select_nonce": 0,
        "related_view_return_target": "main",
        "current_related_view_bfs_mode": DEFAULT_RELATED_VIEW_BFS_MODE,
    }


def _attribution_view_defaults() -> dict[str, Any]:
    """Return default session-state values for the attribution view."""
    return {
        "attribution_view_active": False,
        "current_attribution_qa_id": None,
        "current_attribution_query_full_name": None,
        "current_attribution_query_node_id": None,
        "current_attribution_rerun_nonce": 0,
        "attribution_selected_node_id": None,
        "attribution_pending_node_id": None,
        "attribution_selected_source_key": None,
        "attribution_pending_source_key": None,
        "attribution_plot_select_nonce": 0,
    }


def _reset_related_view_state() -> None:
    """Reset related-variable view state in the current Streamlit session."""
    for key, value in _related_view_defaults().items():
        st.session_state[key] = value


def _reset_attribution_view_state() -> None:
    """Reset attribution view state in the current Streamlit session."""
    for key, value in _attribution_view_defaults().items():
        st.session_state[key] = value


def init_session_state() -> None:
    """Initialize app-level session keys once per browser session.

    Args:
        None.

    Returns:
        `None`:
            This function populates missing Streamlit session-state defaults.
    """
    defaults = {
        "top_graph_select": None,
        "current_graph_id": None,
        "current_highlight_node_id": None,
        "qa_filter": DEFAULT_QA_FILTER,
        "current_expanded_subgraph_id": None,
        "current_subgraph_highlight_var": None,
        "current_subgraph_highlight_edge": None,
        "current_highlight_mode": "neighbor",
        "pending_sidebar_qa_id": None,
        "last_applied_top_wrong_qa": None,
        "pending_jump_graph_id": None,
        "pending_jump_qa_id": None,
        **_related_view_defaults(),
        **_attribution_view_defaults(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_view_state_for_graph_change(new_graph_id: str | None) -> None:
    """Reset volatile UI state when the selected graph changes.

    Args:
        new_graph_id (`str | None`):
            Graph ID that should become the new active selection.

    Returns:
        `None`:
            This function clears graph-specific UI state in `st.session_state`.
    """
    previous_graph_id = st.session_state.get("current_graph_id")
    if previous_graph_id == new_graph_id:
        return

    pending_jump_graph_id = st.session_state.get("pending_jump_graph_id")
    is_query_driven_jump = bool(pending_jump_graph_id and pending_jump_graph_id == new_graph_id)

    st.session_state["current_graph_id"] = new_graph_id
    st.session_state["current_highlight_node_id"] = None
    st.session_state["current_expanded_subgraph_id"] = None
    st.session_state["current_subgraph_highlight_var"] = None
    st.session_state["current_subgraph_highlight_edge"] = None
    st.session_state["pending_sidebar_qa_id"] = None
    if is_query_driven_jump:
        st.session_state["qa_filter"] = "wrong"
    else:
        st.session_state["top_wrong_qa_select"] = ""
        st.session_state["last_applied_top_wrong_qa"] = None
        st.session_state["pending_jump_graph_id"] = None
        st.session_state["pending_jump_qa_id"] = None
    _reset_related_view_state()
    _reset_attribution_view_state()
    if not is_query_driven_jump:
        st.session_state["qa_filter"] = DEFAULT_QA_FILTER
    st.session_state["current_highlight_mode"] = "neighbor"


def set_highlight_node(node_id: str | None, *, mode: str = "node_only") -> None:
    """Update the currently highlighted node in the timeline view.

    Args:
        node_id (`str | None`):
            Node ID to highlight, or `None` to clear the selection.
        mode (`str`, defaults to `"node_only"`):
            Highlight mode used by the timeline view.

    Returns:
        `None`:
            This function mutates `st.session_state` in place.
    """
    st.session_state["current_highlight_node_id"] = node_id
    st.session_state["current_highlight_mode"] = mode


def set_expanded_subgraph(edge_id: str | None) -> None:
    """Select the macro edge whose detail subgraph should be shown.

    Args:
        edge_id (`str | None`):
            Timeline edge ID in `<source_id>-><target_id>` format, or `None`
            to clear the current selection.

    Returns:
        `None`:
            This function mutates `st.session_state` in place.
    """
    st.session_state["current_expanded_subgraph_id"] = edge_id
    # Reset subgraph highlight when edge context changes.
    st.session_state["current_subgraph_highlight_var"] = None
    st.session_state["current_subgraph_highlight_edge"] = None


def activate_related_view(root_node_id: str, *, return_target: str = "main") -> None:
    """Open the related-variable view focused on one root node.

    Args:
        root_node_id (`str`):
            Runtime root node ID for the related-variable view.
        return_target (`str`, defaults to `"main"`):
            UI return target used when closing the related view.

    Returns:
        `None`:
            This function mutates `st.session_state` in place.
    """
    st.session_state["current_related_view_root_var"] = root_node_id
    st.session_state["related_view_node_stack"] = [root_node_id]
    st.session_state["related_view_selected_node_id"] = None
    st.session_state["related_view_pending_node_id"] = None
    st.session_state["related_view_plot_select_nonce"] = 0
    st.session_state["related_view_return_target"] = return_target
    st.session_state["related_view_active"] = True


def ensure_related_view_stack(root_node_id: str) -> list[str]:
    """Normalize the related-view stack around the current root node.

    Args:
        root_node_id (`str`):
            Root node that should remain at the top of the related-view state.

    Returns:
        `list[str]`:
            Normalized stack stored back into `st.session_state`.
    """
    stack_obj = st.session_state.get("related_view_node_stack")
    node_stack = list(stack_obj) if isinstance(stack_obj, list) else []
    if not node_stack or node_stack[-1] != root_node_id:
        node_stack = [root_node_id]
        st.session_state["related_view_node_stack"] = node_stack
    return node_stack


def step_back_related_view() -> bool:
    """Move the related-variable view back to the previous root node.

    Returns:
        `bool`:
            Whether the related-view stack changed.
    """
    stack_obj = st.session_state.get("related_view_node_stack")
    node_stack = list(stack_obj) if isinstance(stack_obj, list) else []
    if len(node_stack) <= 1:
        return False

    node_stack.pop()
    st.session_state["related_view_node_stack"] = node_stack
    st.session_state["current_related_view_root_var"] = node_stack[-1]
    st.session_state["related_view_selected_node_id"] = None
    st.session_state["related_view_pending_node_id"] = None
    st.session_state["related_view_plot_select_nonce"] = int(
        st.session_state.get("related_view_plot_select_nonce", 0)
    ) + 1
    return True


def close_related_view() -> None:
    """Close the related-variable view and restore the previous overlay target."""
    return_target = str(st.session_state.get("related_view_return_target", "main"))
    _reset_related_view_state()
    if return_target == "attribution":
        st.session_state["attribution_view_active"] = True


def push_related_view_node(node_id: str) -> None:
    """Promote one clicked node as the new related-view root.

    Args:
        node_id (`str`):
            Node that should become the next related-view root.

    Returns:
        `None`:
            This function mutates `st.session_state` in place.
    """
    current_root = st.session_state.get("current_related_view_root_var")
    node_stack = ensure_related_view_stack(str(current_root))
    node_stack.append(node_id)
    st.session_state["related_view_node_stack"] = node_stack
    st.session_state["current_related_view_root_var"] = node_id
    st.session_state["related_view_pending_node_id"] = None
    st.session_state["related_view_plot_select_nonce"] = int(
        st.session_state.get("related_view_plot_select_nonce", 0)
    ) + 1


def set_attribution_target(
    qa_id: str | None,
    query_full_name: str | None,
    query_node_id: str | None,
) -> None:
    """Open or clear the error-attribution overlay target.

    Args:
        qa_id (`str | None`):
            QA record identifier, or `None` to close attribution view.
        query_full_name (`str | None`):
            Full query variable name for the selected QA item.
        query_node_id (`str | None`):
            Full query node ID for the selected QA item.

    Returns:
        `None`:
            This function mutates `st.session_state` in place.
    """
    st.session_state["attribution_view_active"] = qa_id is not None
    st.session_state["current_attribution_qa_id"] = qa_id
    st.session_state["current_attribution_query_full_name"] = query_full_name
    st.session_state["current_attribution_query_node_id"] = query_node_id
    st.session_state["current_attribution_rerun_nonce"] = 0
    if qa_id is not None:
        # Attribution view occupies the same expanded area as related-view.
        _reset_related_view_state()


def close_attribution_view() -> None:
    """Close the attribution overlay and clear its transient selection state."""
    st.session_state["attribution_view_active"] = False
    st.session_state["current_attribution_qa_id"] = None
    st.session_state["current_attribution_query_full_name"] = None
    st.session_state["current_attribution_query_node_id"] = None
    st.session_state["current_attribution_rerun_nonce"] = 0
    st.session_state["attribution_selected_node_id"] = None
    st.session_state["attribution_pending_node_id"] = None
    st.session_state["attribution_selected_source_key"] = None
    st.session_state["attribution_pending_source_key"] = None
    st.session_state["attribution_plot_select_nonce"] = 0
