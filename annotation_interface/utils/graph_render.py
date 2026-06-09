"""Shared graph rendering and Plotly interaction helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx


@dataclass(slots=True)
class EdgeRenderGeometry:
    """Precomputed geometry used to render one interaction edge."""

    path_points: list[tuple[float, float]]
    arrow_end: tuple[float, float]
    arrow_angle: float
    midpoint: tuple[float, float] | None
    hit_points: list[tuple[float, float]]


def calculate_arrow_angle(dx: float, dy: float) -> float:
    """Return the Plotly marker angle for an edge arrowhead.

    Args:
        dx (`float`):
            Horizontal edge displacement.
        dy (`float`):
            Vertical edge displacement.

    Returns:
        `float`:
            Angle in degrees expected by Plotly marker rotation.
    """
    radians = math.atan2(dy, dx)
    degrees = math.degrees(radians)
    return degrees + 90


def arrow_symbol(dx: float, dy: float) -> str:
    """Choose the Plotly marker symbol that best matches an edge direction.

    Args:
        dx (`float`):
            Horizontal edge displacement.
        dy (`float`):
            Vertical edge displacement.

    Returns:
        `str`:
            Plotly marker symbol name for the arrowhead.
    """
    if abs(dx) >= abs(dy):
        return "triangle-right" if dx >= 0 else "triangle-left"
    return "triangle-down" if dy >= 0 else "triangle-up"


def color_with_alpha(color: str, alpha: float) -> str:
    """Apply a new alpha value to one CSS color string.

    Args:
        color (`str`):
            CSS color string in `hsl`, `rgb`, or `rgba` form.
        alpha (`float`):
            Desired opacity value.

    Returns:
        `str`:
            Color string with the requested alpha channel.
    """
    if color.startswith("hsl("):
        return color.replace("hsl(", "hsla(", 1).replace(")", f", {alpha})", 1)
    if color.startswith("rgb("):
        return f"rgba({color[4:-1]}, {alpha})"
    if color.startswith("rgba("):
        parts = [p.strip() for p in color[5:-1].split(",")]
        if len(parts) >= 3:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    return color


def build_ordered_layout(graph: nx.DiGraph, created_at_by_node: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Generate a layered layout ordered by dependency depth and timestamp.

    Args:
        graph (`nx.DiGraph`):
            Directed graph whose nodes should be positioned.
        created_at_by_node (`dict[str, str]`):
            Mapping from node ID to creation timestamp string used for stable ordering.

    Returns:
        `dict[str, tuple[float, float]]`:
            Node positions keyed by node ID.
    """
    if not graph.nodes:
        return {}

    indegree_left = {n: graph.in_degree(n) for n in graph.nodes}
    level: dict[str, int] = {}
    queue = [n for n, deg in indegree_left.items() if deg == 0]
    queue.sort(key=lambda n: (created_at_by_node.get(n, ""), n))

    for n in queue:
        level[n] = 0

    while queue:
        u = queue.pop(0)
        for v in graph.successors(u):
            level[v] = max(level.get(v, 0), level[u] + 1)
            indegree_left[v] -= 1
            if indegree_left[v] == 0:
                queue.append(v)

    unresolved = [n for n in graph.nodes if n not in level]
    for n in sorted(unresolved, key=lambda x: (created_at_by_node.get(x, ""), x)):
        parent_levels = [level[p] for p in graph.predecessors(n) if p in level]
        level[n] = (max(parent_levels) + 1) if parent_levels else 0

    layer_to_nodes: dict[int, list[str]] = {}
    for n, lv in level.items():
        layer_to_nodes.setdefault(lv, []).append(n)

    pos: dict[str, tuple[float, float]] = {}
    for lv in sorted(layer_to_nodes):
        layer_nodes = sorted(layer_to_nodes[lv], key=lambda n: (created_at_by_node.get(n, ""), n))
        count = len(layer_nodes)
        center = (count - 1) / 2.0
        for idx, node_id in enumerate(layer_nodes):
            x = float(lv * 3.6)
            y = float((idx - center) * -2.4)
            pos[node_id] = (x, y)

    return pos


def compact_label_from_full_node_id(full_node_id: str) -> str:
    """Shorten long node IDs while preserving readable prefixes and suffixes.

    Args:
        full_node_id (`str`):
            Full node identifier shown in the UI.

    Returns:
        `str`:
            Compacted node label suitable for figure text.
    """
    full = str(full_node_id)
    compact = re.sub(r":([0-9a-fA-F]{16,})(@\d+)$", r":\2", full)
    compact = re.sub(r"^(memory-unit)-[0-9a-fA-F-]{8,}(@\d+)$", r"\1\2", compact)
    return compact


def format_hover_value(value: Any) -> str:
    """Format one value for display inside a hover tooltip.

    Args:
        value (`Any`):
            Raw value that should be rendered in hover text.

    Returns:
        `str`:
            Stringified and length-limited tooltip text.
    """
    if isinstance(value, (dict, list)):
        val_str = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        val_str = str(value)

    max_total_len = 1000
    if len(val_str) > max_total_len:
        val_str = val_str[:max_total_len] + "... (Value truncated)"

    return val_str


def wrap_hover_text(text: str, width: int = 80) -> str:
    """Wrap tooltip text into multiple lines for readable hover cards.

    Args:
        text (`str`):
            Raw tooltip text.
        width (`int`, defaults to `80`):
            Maximum wrapped line width.

    Returns:
        `str`:
            Tooltip text joined with HTML line breaks.
    """
    lines = str(text).splitlines() or [""]
    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=True, break_on_hyphens=False))
    return "<br>".join(wrapped_lines)


def build_runtime_node_hover(
    *,
    full_node_id: str,
    full_name: str,
    category: str,
    class_name: str,
    comment: str,
    value: Any,
) -> str:
    """Build the shared hover card text for one runtime variable node.

    Args:
        full_node_id (`str`):
            Runtime full node ID shown in the tooltip.
        full_name (`str`):
            Fully qualified variable name.
        category (`str`):
            Normalized node category.
        class_name (`str`):
            Runtime class name attached to the variable.
        comment (`str`):
            Runtime node comment text.
        value (`Any`):
            Runtime node value shown in the tooltip.

    Returns:
        `str`:
            HTML hover text used by Plotly node traces.
    """
    return (
        f"<br><b>full_node_id</b>: {wrap_hover_text(full_node_id, width=80)}"
        + f"<br><b>full_name</b>: {wrap_hover_text(full_name, width=80)}"
        + f"<br><b>category</b>: {category}"
        + f"<br><b>class_name</b>: {class_name}"
        + f"<br><b>comment</b>: {wrap_hover_text(comment, width=70)}"
        + f"<br><b>value</b>: {wrap_hover_text(format_hover_value(value), width=90)}"
    )


def build_runtime_edge_hover(
    *,
    edge_id: str,
    op_id: str,
    category: str,
    source_id: str,
    target_id: str,
    comment: str,
    created_at: str | None = None,
    source_label: str = "source_id",
    target_label: str = "target_id",
) -> str:
    """Build the shared hover card text for one runtime dependency edge.

    Args:
        edge_id (`str`):
            Runtime edge identifier.
        op_id (`str`):
            Runtime operation identifier.
        category (`str`):
            Runtime edge category.
        source_id (`str`):
            Source full node ID.
        target_id (`str`):
            Target full node ID.
        comment (`str`):
            Runtime edge comment text.
        created_at (`str | None`, optional):
            Runtime edge creation timestamp when it should be displayed.
        source_label (`str`, defaults to `"source_id"`):
            Display label for the source endpoint field.
        target_label (`str`, defaults to `"target_id"`):
            Display label for the target endpoint field.

    Returns:
        `str`:
            HTML hover text used by Plotly edge traces.
    """
    hover = f"<b>edge_id</b>: {edge_id}" + f"<br><b>op_id</b>: {op_id}"
    if created_at:
        hover += f"<br><b>created_at</b>: {created_at}"
    hover += (
        f"<br><b>category</b>: {category}"
        + f"<br><b>{source_label}</b>: {wrap_hover_text(source_id, width=70)}"
        + f"<br><b>{target_label}</b>: {wrap_hover_text(target_id, width=70)}"
        + f"<br><b>comment</b>: {wrap_hover_text(comment, width=80)}"
    )
    return hover


def build_runtime_node_render_attrs(node: Any) -> dict[str, Any]:
    """Build shared render attributes for one runtime variable node.

    Args:
        node (`Any`):
            Runtime variable object returned by smartcomment.

    Returns:
        `dict[str, Any]`:
            Common node render fields including category, label, color, hover,
            and creation timestamp.
    """
    return {
        "category": node.category,
        "label": compact_label_from_full_node_id(node.full_node_id),
        "color": node_color_from_category(node.category),
        "hover": build_runtime_node_hover(
            full_node_id=node.full_node_id,
            full_name=node.full_name,
            category=node.category,
            class_name=node.class_name,
            comment=node.comment,
            value=node.value,
        ),
        "created_at": node.created_at,
    }


def build_runtime_edge_render_attrs(edge: Any, *, include_created_at: bool = False) -> dict[str, Any]:
    """Build shared render attributes for one runtime dependency edge.

    Args:
        edge (`Any`):
            Runtime edge object returned by smartcomment.
        include_created_at (`bool`, defaults to `False`):
            Whether the hover tooltip should include the edge timestamp.

    Returns:
        `dict[str, Any]`:
            Common edge render fields including normalized op ID, color, and hover.
    """
    return {
        "op_id": edge.op_id,
        "color": edge_color_from_op_id(edge.op_id),
        "hover": build_runtime_edge_hover(
            edge_id=edge.edge_id,
            op_id=edge.op_id,
            category=edge.category,
            source_id=edge.source_full_node_id,
            target_id=edge.target_full_node_id,
            comment=edge.comment,
            created_at=edge.created_at,
        ),
    }


def is_point_near_any_node(px: float, py: float, node_points: list[tuple[float, float]], threshold: float = 0.35) -> bool:
    """Check whether one point is too close to any node center.

    Args:
        px (`float`):
            X coordinate to test.
        py (`float`):
            Y coordinate to test.
        node_points (`list[tuple[float, float]]`):
            Node center coordinates.
        threshold (`float`, defaults to `0.35`):
            Distance threshold below which the point is considered overlapping.

    Returns:
        `bool`:
            Whether the test point is near any node center.
    """
    for nx, ny in node_points:
        dx = px - nx
        dy = py - ny
        if (dx * dx + dy * dy) ** 0.5 <= threshold:
            return True
    return False


def _matplotlib_color_from_key(key: str, cmap_name: str = "tab20") -> str:
    """Map one stable string key to a deterministic Matplotlib color.

    Args:
        key (`str`):
            Stable string key used to derive the color hash.
        cmap_name (`str`, defaults to `"tab20"`):
            Matplotlib colormap name used for color lookup.

    Returns:
        `str`:
            Hex color string produced from the target colormap.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    hash_int = int(digest[:16], 16)
    cmap_obj = plt.get_cmap(cmap_name)
    steps = 256
    idx = hash_int % steps
    rgba = cmap_obj(idx / max(1, steps - 1))
    return mcolors.to_hex(rgba)


def node_color_from_category(category: str | None) -> str:
    """Map node category to a deterministic Matplotlib color.

    Args:
        category (`str | None`):
            Raw node category from parsed graph data.

    Returns:
        `str`:
            CSS color string used by the frontend renderers.
    """
    normalized = str(category or "").strip() or "unknown"
    return _matplotlib_color_from_key(normalized.lower())


def edge_color_from_op_id(op_id: str | None) -> str:
    """Map one operation ID to a deterministic Matplotlib color.

    Args:
        op_id (`str | None`):
            Operation identifier attached to one runtime edge.

    Returns:
        `str`:
            CSS color string derived from the operation ID hash.
    """
    normalized = str(op_id or "").strip() or "unknown_op"
    return _matplotlib_color_from_key(normalized)


def resample_polyline(points: list[tuple[float, float]], samples: int) -> list[tuple[float, float]]:
    """Uniformly sample points along a polyline by arc length.

    Args:
        points (`list[tuple[float, float]]`):
            Polyline vertices in order.
        samples (`int`):
            Number of points to sample along the polyline.

    Returns:
        `list[tuple[float, float]]`:
            Resampled polyline points distributed by arc length.
    """
    if not points:
        return []
    if len(points) == 1 or samples <= 1:
        return [points[0]]

    cumulative: list[float] = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        total += seg
        cumulative.append(total)

    if total <= 1e-9:
        return [points[0] for _ in range(samples)]

    out: list[tuple[float, float]] = []
    for k in range(samples):
        target = total * (k / (samples - 1))
        seg_idx = 0
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < target:
            seg_idx += 1

        start_len = cumulative[seg_idx]
        end_len = cumulative[min(seg_idx + 1, len(cumulative) - 1)]
        x0, y0 = points[seg_idx]
        x1, y1 = points[min(seg_idx + 1, len(points) - 1)]

        if end_len - start_len <= 1e-9:
            out.append((x0, y0))
            continue

        t = (target - start_len) / (end_len - start_len)
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def sample_edge_path(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    node_points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Generate a straight or curved edge path between two rendered nodes.

    Args:
        x0 (`float`):
            Source X coordinate.
        y0 (`float`):
            Source Y coordinate.
        x1 (`float`):
            Target X coordinate.
        y1 (`float`):
            Target Y coordinate.
        node_points (`list[tuple[float, float]]`):
            All node center coordinates used to detect same-row crossings.

    Returns:
        `list[tuple[float, float]]`:
            Polyline points describing the edge path.
    """
    is_same_row = abs(y0 - y1) < 1e-6
    span_x = abs(x1 - x0)

    if not is_same_row:
        return [(x0, y0), (x1, y1)]

    min_x = min(x0, x1)
    max_x = max(x0, x1)
    has_intermediate_node = False
    for nx, ny in node_points:
        if abs(ny - y0) > 1e-6:
            continue
        if nx <= min_x + 1e-6 or nx >= max_x - 1e-6:
            continue
        has_intermediate_node = True
        break

    if not has_intermediate_node:
        return [(x0, y0), (x1, y1)]

    ctrl_x = (x0 + x1) / 2
    lift = min(2.2, max(0.9, span_x * 0.22))
    ctrl_y = y0 - lift

    points: list[tuple[float, float]] = []
    samples = 17
    for i in range(samples):
        t = i / (samples - 1)
        omt = 1 - t
        x = omt * omt * x0 + 2 * omt * t * ctrl_x + t * t * x1
        y = omt * omt * y0 + 2 * omt * t * ctrl_y + t * t * y1
        points.append((x, y))
    return points


def build_edge_render_geometry(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    node_points: list[tuple[float, float]],
    *,
    samples: int = 25,
    endpoint_trim_ratio: float = 0.08,
) -> EdgeRenderGeometry:
    """Build sampled geometry for one rendered edge.

    Args:
        x0 (`float`):
            Source X coordinate.
        y0 (`float`):
            Source Y coordinate.
        x1 (`float`):
            Target X coordinate.
        y1 (`float`):
            Target Y coordinate.
        node_points (`list[tuple[float, float]]`):
            All rendered node centers used to avoid hover points on nodes.
        samples (`int`, defaults to `25`):
            Number of resampled points used for arrow and hover geometry.
        endpoint_trim_ratio (`float`, defaults to `0.08`):
            Portion of the sampled path trimmed away near both endpoints.

    Returns:
        `EdgeRenderGeometry`:
            Shared edge geometry used by different graph views.
    """
    path_points = sample_edge_path(x0, y0, x1, y1, node_points)
    sampled_path = resample_polyline(path_points, samples=samples)

    end_x, end_y = sampled_path[-1]
    prev_x, prev_y = sampled_path[-2] if len(sampled_path) >= 2 else (x0, y0)
    arrow_angle = calculate_arrow_angle(end_x - prev_x, end_y - prev_y)

    midpoint: tuple[float, float] | None = None
    mid_idx = len(sampled_path) // 2
    mid_x, mid_y = sampled_path[mid_idx]
    if not is_point_near_any_node(mid_x, mid_y, node_points):
        midpoint = (mid_x, mid_y)

    hit_points: list[tuple[float, float]] = []
    total_samples = len(sampled_path)
    for i, (hx, hy) in enumerate(sampled_path):
        t = i / (total_samples - 1) if total_samples > 1 else 0.0
        if t <= endpoint_trim_ratio or t >= 1.0 - endpoint_trim_ratio:
            continue
        if is_point_near_any_node(hx, hy, node_points):
            continue
        hit_points.append((hx, hy))

    return EdgeRenderGeometry(
        path_points=path_points,
        arrow_end=(end_x, end_y),
        arrow_angle=arrow_angle,
        midpoint=midpoint,
        hit_points=hit_points,
    )


def extract_selected_points(event: Any) -> list[dict[str, Any]]:
    """Extract selected Plotly points from a Streamlit event payload.

    Args:
        event (`Any`):
            Plotly selection payload returned by `st.plotly_chart`.

    Returns:
        `list[dict[str, Any]]`:
            Selected point payloads, or an empty list when nothing is selected.
    """
    if not event:
        return []
    if isinstance(event, dict):
        selection = event.get("selection") or {}
        points = selection.get("points")
        if isinstance(points, list):
            return points
    return []


def parse_customdata(raw_customdata: Any) -> tuple[str, str] | None:
    """Parse `<kind>|<payload>` values from Plotly `customdata`.

    Args:
        raw_customdata (`Any`):
            Raw `customdata` payload emitted by Plotly.

    Returns:
        `tuple[str, str] | None`:
            Parsed `(kind, payload)` tuple, or `None` when the payload format
            is invalid.
    """
    if raw_customdata is None:
        return None

    if isinstance(raw_customdata, list) and raw_customdata:
        value = str(raw_customdata[0])
    else:
        value = str(raw_customdata)

    if "|" not in value:
        return None
    kind, payload = value.split("|", 1)
    return kind, payload


def pick_interaction_payload(
    points: list[dict[str, Any]],
    preferred_kinds: list[str],
) -> tuple[str, str] | None:
    """Pick one deterministic payload from overlapping Plotly selection points.

    Args:
        points (`list[dict[str, Any]]`):
            Selected Plotly point payloads returned by Streamlit.
        preferred_kinds (`list[str]`):
            Payload kinds in descending priority order.

    Returns:
        `tuple[str, str] | None`:
            Parsed `(kind, payload)` tuple for the first matching preferred kind,
            or `None` when no supported payload is present.
    """
    parsed: list[tuple[str, str]] = []
    for point in points:
        payload = parse_customdata(point.get("customdata"))
        if payload:
            parsed.append(payload)

    for preferred_kind in preferred_kinds:
        for kind, value in parsed:
            if kind == preferred_kind:
                return kind, value
    return None


def pick_clicked_node_id(points: list[dict[str, Any]]) -> str | None:
    """Return the clicked node ID from a Plotly selection payload.

    Args:
        points (`list[dict[str, Any]]`):
            Selected Plotly point payloads.

    Returns:
        `str | None`:
            The selected node ID, or `None` when the click did not target a node.
    """
    payload = pick_interaction_payload(points, preferred_kinds=["node"])
    if payload is None:
        return None
    _, node_id = payload
    return node_id or None
