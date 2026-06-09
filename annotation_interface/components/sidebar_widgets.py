"""Sidebar controls for graph/QA filtering and evidence targeting."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from data_engine.parser import GraphRecord
from utils.session_state import (
    QA_FILTER_ALL,
    QA_FILTER_CORRECT,
    QA_FILTER_WRONG,
    set_attribution_target,
    set_highlight_node,
)


@dataclass(slots=True)
class QAItem:
    qa_id: str
    prediction_text: str
    golden_text: str
    is_correct: bool
    question_text: str
    query_full_name: str | None
    query_node_id: str | None
    evidence_highlight_ids: list[str | None]
    evidence_texts: list[str]


def _render_qa_card(index: int, item: QAItem, status_text: str, status_class: str) -> None:
    """Render one QA detail card in the sidebar.

    Args:
        index (`int`):
            One-based display index within the filtered QA list.
        item (`QAItem`):
            QA item to render.
        status_text (`str`):
            Human-readable correctness label.
        status_class (`str`):
            CSS modifier class used by the status badge.

    Returns:
        `None`:
            This function renders directly into the current Streamlit container.
    """
    evidence_preview = "<br>".join(
        f"{idx + 1}. {text}"
        for idx, text in enumerate(item.evidence_texts)
    ) if item.evidence_texts else "(No evidence resolved)"
    st.markdown(
        (
            "<div class='qa-card'>"
            f"<div class='qa-card-head'><span class='qa-no'>Q{index}</span>"
            f"<span class='qa-status {status_class}'>{status_text}</span></div>"
            "<div class='qa-label'>Query</div>"
            f"<div class='qa-q'>{item.question_text}</div>"
            "<div class='qa-label'>Query Full Name</div>"
            f"<div class='qa-a'>{item.query_full_name or '(N/A)'}</div>"
            "<div class='qa-label'>Prediction</div>"
            f"<div class='qa-a'>{item.prediction_text}</div>"
            "<div class='qa-label'>Golden Answer</div>"
            f"<div class='qa-a'>{item.golden_text}</div>"
            "<div class='qa-label'>Evidence</div>"
            f"<div class='qa-a'>{evidence_preview}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_qa_items_from_protocol(graph: GraphRecord) -> list[QAItem]:
    """Convert parsed QA records into sidebar view models.

    Args:
        graph (`GraphRecord`):
            Parsed graph record for the currently selected trace.

    Returns:
        `list[QAItem]`:
            Sidebar-ready QA items derived from `graph.qa_lists`.
    """
    return [
        QAItem(
            qa_id=qa.qa_id,
            prediction_text=qa.predicted_answer,
            golden_text=qa.golden_answer,
            is_correct=qa.is_correct,
            question_text=qa.question,
            query_full_name=qa.query_full_name,
            query_node_id=qa.query_node_id,
            evidence_highlight_ids=qa.source_evidence_ids,
            evidence_texts=qa.source_evidence_texts,
        )
        for qa in graph.qa_lists
    ]


def render_sidebar_widgets(graph: GraphRecord | None) -> None:
    """Render QA-only sidebar controls for the selected graph.

    Args:
        graph (`GraphRecord | None`):
            Parsed graph record currently selected in the main app, or `None`
            when no graph is available.

    Returns:
        `None`:
            This function renders sidebar widgets into the active Streamlit page.
    """
    # st.subheader("Q&A")
    if graph is None:
        st.info("No graph selected.")
        return

    st.markdown("### QA Filter")
    label_to_value = {
        "All": QA_FILTER_ALL,
        "Correct": QA_FILTER_CORRECT,
        "Wrong": QA_FILTER_WRONG,
    }
    current_filter = st.session_state.get("qa_filter", QA_FILTER_ALL)
    selected_filter_label = st.radio(
        "Filter questions",
        options=list(label_to_value.keys()),
        index=list(label_to_value.values()).index(current_filter),
        horizontal=True,
    )
    st.session_state["qa_filter"] = label_to_value[selected_filter_label]

    qa_items = _build_qa_items_from_protocol(graph)
    st.markdown(
        f"""
        <div style='display: flex; align-items: baseline; gap: 10px; margin-bottom: -15px;'>
            <h3 style='margin: 0;'>QA List</h3>
            <span style='color: gray; font-size: 0.8rem;'>Total QA pairs: {len(qa_items)}</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    filtered_items = qa_items
    if st.session_state["qa_filter"] == QA_FILTER_CORRECT:
        filtered_items = [item for item in qa_items if item.is_correct]
    elif st.session_state["qa_filter"] == QA_FILTER_WRONG:
        filtered_items = [item for item in qa_items if not item.is_correct]

    if not filtered_items:
        st.caption("There are no QAs under the current filtering conditions.")
        return

    qa_options: list[tuple[str, QAItem]] = []
    for idx, item in enumerate(filtered_items):
        display_name = item.query_full_name or item.question_text
        qa_options.append((f"{idx + 1}. {display_name}", item))

    option_labels = [label for label, _ in qa_options]
    selector_key = f"qa_selector_{graph.graph_id}_{st.session_state['qa_filter']}"
    pending_sidebar_qa_id = st.session_state.get("pending_sidebar_qa_id")
    if pending_sidebar_qa_id:
        pending_label = next(
            (label for label, item in qa_options if item.qa_id == pending_sidebar_qa_id),
            None,
        )
        if pending_label is not None:
            st.session_state[selector_key] = pending_label

    selected_label = st.selectbox(
        "Select QA",
        options=option_labels,
        key=selector_key,
    )
    selected_item = next(item for label, item in qa_options if label == selected_label)
    if pending_sidebar_qa_id and selected_item.qa_id == pending_sidebar_qa_id:
        st.session_state["pending_sidebar_qa_id"] = None

    with st.container():
        st.markdown("<div class='qa-scroll-host'>", unsafe_allow_html=True)
        # with _qa_scroll_container():
        with st.container(height=560, border=True):
            status_text = "Correct" if selected_item.is_correct else "Wrong"
            status_class = "ok" if selected_item.is_correct else "bad"
            selected_idx = option_labels.index(selected_label)
            _render_qa_card(selected_idx + 1, selected_item, status_text, status_class)

            if not selected_item.is_correct:
                if st.button(
                    "Run Error Attribution",
                    key=f"attr_btn_{graph.graph_id}_{selected_item.qa_id}",
                    use_container_width=False,
                ):
                    set_attribution_target(
                        qa_id=selected_item.qa_id,
                        query_full_name=selected_item.query_full_name,
                        query_node_id=selected_item.query_node_id,
                    )
                    st.rerun()

            evidence_ids = selected_item.evidence_highlight_ids
            if not evidence_ids:
                st.caption("No source evidence.")
            else:
                button_cols = st.columns(min(4, len(evidence_ids)))
                for ev_idx, ev_node_id in enumerate(evidence_ids):
                    col = button_cols[ev_idx % len(button_cols)]
                    with col:
                        btn_disabled = ev_node_id is None
                        if st.button(
                            f"Evidence {ev_idx + 1}",
                            key=f"evidence_btn_{graph.graph_id}_{selected_item.qa_id}_{ev_idx}",
                            disabled=btn_disabled,
                            use_container_width=True,
                        ):
                            set_highlight_node(ev_node_id, mode="node_only")
        st.markdown("</div>", unsafe_allow_html=True)
