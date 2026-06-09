"""Geometry helpers for timeline and session visualization.

This module provides:
1) Boustrophedon (serpentine) coordinates for message nodes.
2) Consecutive node edges with row-turn (vertical connector) detection.
3) Session cover rectangle computation.
4) Session orthogonal polygon path generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import HORIZONTAL_SPACING, NODES_PER_ROW, VERTICAL_SPACING

@dataclass(frozen=True)
class Point:
    """2D point in canvas coordinates."""
    x: float
    y: float

@dataclass(frozen=True)
class Edge:
    """Connection between two consecutive nodes.
    `is_row_connector` is True when the edge is vertical and indicates row-to-row turn.
    """
    src_index: int
    dst_index: int
    src: Point
    dst: Point
    is_row_connector: bool

@dataclass
class Rect:
    """Helper to store rectangle bounds."""
    x: float
    y: float
    w: float
    h: float

def _index_to_boustrophedon_xy(
    index: int,
    *,
    nodes_per_row: int = NODES_PER_ROW,
    horizontal_spacing: float = HORIZONTAL_SPACING,
    vertical_spacing: float = VERTICAL_SPACING,
) -> Point:
    """Map one linear node index to boustrophedon canvas coordinates.

    Args:
        index (`int`):
            Zero-based node index in timeline order.
        nodes_per_row (`int`, defaults to `NODES_PER_ROW`):
            Number of nodes shown in each visual row.
        horizontal_spacing (`float`, defaults to `HORIZONTAL_SPACING`):
            Horizontal spacing between adjacent columns.
        vertical_spacing (`float`, defaults to `VERTICAL_SPACING`):
            Vertical spacing between adjacent rows.

    Returns:
        `Point`:
            Canvas coordinate for the indexed node.
    """
    if index < 0:
        raise ValueError("index must be >= 0")
    if nodes_per_row <= 0:
        raise ValueError("nodes_per_row must be > 0")

    row = index // nodes_per_row
    col_in_row = index % nodes_per_row
    # Core serpentine mapping: odd rows mirror the visual column.
    visual_col = col_in_row if row % 2 == 0 else (nodes_per_row - 1 - col_in_row)

    x = visual_col * horizontal_spacing
    y = row * vertical_spacing
    return Point(x=x, y=y)

def build_node_positions(
    num_nodes: int,
    *,
    nodes_per_row: int = NODES_PER_ROW,
    horizontal_spacing: float = HORIZONTAL_SPACING,
    vertical_spacing: float = VERTICAL_SPACING,
) -> dict[int, Point]:
    """Build canvas positions for all node indices in one timeline.

    Args:
        num_nodes (`int`):
            Number of nodes that should be positioned.
        nodes_per_row (`int`, defaults to `NODES_PER_ROW`):
            Number of nodes shown in each visual row.
        horizontal_spacing (`float`, defaults to `HORIZONTAL_SPACING`):
            Horizontal spacing between adjacent columns.
        vertical_spacing (`float`, defaults to `VERTICAL_SPACING`):
            Vertical spacing between adjacent rows.

    Returns:
        `dict[int, Point]`:
            Position mapping keyed by zero-based node index.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be >= 0")

    return {
        i: _index_to_boustrophedon_xy(
            i,
            nodes_per_row=nodes_per_row,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
        for i in range(num_nodes)
    }

def build_consecutive_edges(
    num_nodes: int,
    *,
    nodes_per_row: int = NODES_PER_ROW,
    horizontal_spacing: float = HORIZONTAL_SPACING,
    vertical_spacing: float = VERTICAL_SPACING,
) -> list[Edge]:
    """Create edges between consecutive nodes in timeline order.

    Args:
        num_nodes (`int`):
            Number of nodes in the timeline.
        nodes_per_row (`int`, defaults to `NODES_PER_ROW`):
            Number of nodes shown in each visual row.
        horizontal_spacing (`float`, defaults to `HORIZONTAL_SPACING`):
            Horizontal spacing between adjacent columns.
        vertical_spacing (`float`, defaults to `VERTICAL_SPACING`):
            Vertical spacing between adjacent rows.

    Returns:
        `list[Edge]`:
            Consecutive timeline edges with row-turn markers.
    """
    if num_nodes <= 1:
        return []

    positions = build_node_positions(
        num_nodes,
        nodes_per_row=nodes_per_row,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    edges: list[Edge] = []
    for i in range(num_nodes - 1):
        src = positions[i]
        dst = positions[i + 1]
        # Row boundary happens at the last column in the linear row traversal.
        is_row_connector = ((i + 1) % nodes_per_row == 0)
        edges.append(
            Edge(
                src_index=i,
                dst_index=i + 1,
                src=src,
                dst=dst,
                is_row_connector=is_row_connector,
            )
        )

    return edges


def clip_edge_ends(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    gap: float = 35.0,
) -> tuple[float, float, float, float]:
    """Shrink one edge line near both node centers.

    Args:
        x0 (`float`):
            Source X coordinate.
        y0 (`float`):
            Source Y coordinate.
        x1 (`float`):
            Target X coordinate.
        y1 (`float`):
            Target Y coordinate.
        gap (`float`, defaults to `35.0`):
            Maximum shortening distance applied at each end.

    Returns:
        `tuple[float, float, float, float]`:
            Clipped `(x0, y0, x1, y1)` coordinates for the rendered segment.
    """
    dx = x1 - x0
    dy = y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-6:
        return x0, y0, x1, y1
    shrink = min(gap, max(0.0, length * 0.35))
    ux = dx / length
    uy = dy / length
    return x0 + ux * shrink, y0 + uy * shrink, x1 - ux * shrink, y1 - uy * shrink


def polygon_points_to_path(points: list[Point]) -> str:
    """Convert polygon vertices into an SVG path string.

    Args:
        points (`list[Point]`):
            Polygon vertices in order.

    Returns:
        `str`:
            SVG path string for `fig.add_shape(type="path", ...)`.
    """
    if not points:
        return ""
    coords = [f"{p.x},{p.y}" for p in points]
    return "M " + " L ".join(coords) + " Z"


def _build_session_cover_rectangles(
    indices: list[int],
    *,
    nodes_per_row: int,
    horizontal_spacing: float,
    vertical_spacing: float,
    padding_x: float,
    padding_y: float,
    connector_width_ratio: float,
) -> list[Rect]:
    """Build overlapping rectangles that cover one session path.

    Args:
        indices (`list[int]`):
            Timeline node indices that belong to the session.
        nodes_per_row (`int`):
            Number of nodes shown in each visual row.
        horizontal_spacing (`float`):
            Horizontal spacing between adjacent columns.
        vertical_spacing (`float`):
            Vertical spacing between adjacent rows.
        padding_x (`float`):
            Horizontal padding ratio applied around row rectangles.
        padding_y (`float`):
            Vertical padding ratio applied around row rectangles.
        connector_width_ratio (`float`):
            Width ratio used for vertical connector rectangles across rows.

    Returns:
        `list[Rect]`:
            Overlapping rectangles whose union covers the session path.
    """
    if not indices:
        return []

    # Group the node indices by row
    rows_map: dict[int, list[int]] = {}
    for idx in indices:
        row = idx // nodes_per_row
        rows_map.setdefault(row, []).append(idx)
    
    rects: list[Rect] = []
    sorted_rows = sorted(rows_map.keys())
    
    # Convert the spacing ratio to the actual pixel value
    pad_x = padding_x * horizontal_spacing
    pad_y = padding_y * vertical_spacing
    conn_half_w = connector_width_ratio * horizontal_spacing

    for i, row in enumerate(sorted_rows):
        # Generate the horizontal covering rectangle for the current row
        row_indices = rows_map[row]
        row_points = [
            _index_to_boustrophedon_xy(
                idx, 
                nodes_per_row=nodes_per_row, 
                horizontal_spacing=horizontal_spacing, 
                vertical_spacing=vertical_spacing
            )
            for idx in row_indices
        ]
        xs = [p.x for p in row_points]
        min_x, max_x = min(xs), max(xs)
        
        current_h_rect = Rect(
            x=min_x - pad_x,
            y=row * vertical_spacing - pad_y,
            w=(max_x - min_x) + 2 * pad_x,
            h=2 * pad_y
        )
        rects.append(current_h_rect)

        # Generate vertical connected rectangles across multiple rows
        if i < len(sorted_rows) - 1:
            next_row = sorted_rows[i + 1]
            
            exit_node_pos = _index_to_boustrophedon_xy(
                row_indices[-1], 
                nodes_per_row=nodes_per_row, 
                horizontal_spacing=horizontal_spacing, 
                vertical_spacing=vertical_spacing
            )
            entry_node_pos = _index_to_boustrophedon_xy(
                rows_map[next_row][0], 
                nodes_per_row=nodes_per_row, 
                horizontal_spacing=horizontal_spacing, 
                vertical_spacing=vertical_spacing
            )

            v_x_min = min(exit_node_pos.x, entry_node_pos.x) - conn_half_w
            v_x_max = max(exit_node_pos.x, entry_node_pos.x) + conn_half_w
            
            connector_rect = Rect(
                x=v_x_min,
                y=row * vertical_spacing,
                w=(v_x_max - v_x_min),
                h=(next_row - row) * vertical_spacing
            )
            rects.append(connector_rect)

    return rects


def _rectangles_to_orthogonal_polygon(
    rectangles: list[Rect],
) -> list[Point]:
    """Convert an axis-aligned rectangle union into one outer polygon.

    Args:
        rectangles (`list[tuple[float, float, float, float]]`):
            Rectangles whose union should be converted into an orthogonal outline.

    Returns:
        `list[Point]`:
            Vertices of the largest outer orthogonal polygon.
    """
    if not rectangles:
        return []

    normalized: list[tuple[float, float, float, float]] = []
    xs: set[float] = set()
    ys: set[float] = set()
    for r in rectangles:
        # Rect(x, y, w, h) -> (x0, y0, x1, y1)
        nx0, ny0 = r.x, r.y
        nx1, ny1 = r.x + r.w, r.y + r.h
        
        if nx1 <= nx0 or ny1 <= ny0:
            continue
        normalized.append((nx0, ny0, nx1, ny1))
        xs.update([nx0, nx1])
        ys.update([ny0, ny1])

    if not normalized:
        return []

    x_list = sorted(xs)
    y_list = sorted(ys)
    if len(x_list) < 2 or len(y_list) < 2:
        return []

    nx = len(x_list) - 1
    ny = len(y_list) - 1
    occupied = [[False for _ in range(nx)] for _ in range(ny)]

    # Rasterize by testing each cell center against rectangle union.
    for j in range(ny):
        cy = (y_list[j] + y_list[j + 1]) / 2.0
        for i in range(nx):
            cx = (x_list[i] + x_list[i + 1]) / 2.0
            for rx0, ry0, rx1, ry1 in normalized:
                if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                    occupied[j][i] = True
                    break

    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def is_occ(i: int, j: int) -> bool:
        """Return whether one rasterized grid cell is occupied.

        Args:
            i (`int`):
                Grid-column index.
            j (`int`):
                Grid-row index.

        Returns:
            `bool`:
                Whether the tested cell lies inside the rectangle union.
        """
        if i < 0 or j < 0 or i >= nx or j >= ny:
            return False
        return occupied[j][i]

    # Build oriented boundary edges from occupied cells.
    for j in range(ny):
        for i in range(nx):
            if not occupied[j][i]:
                continue

            x0 = x_list[i]
            x1 = x_list[i + 1]
            y0 = y_list[j]
            y1 = y_list[j + 1]

            if not is_occ(i, j - 1):
                edges.append(((x0, y0), (x1, y0)))
            if not is_occ(i + 1, j):
                edges.append(((x1, y0), (x1, y1)))
            if not is_occ(i, j + 1):
                edges.append(((x1, y1), (x0, y1)))
            if not is_occ(i - 1, j):
                edges.append(((x0, y1), (x0, y0)))

    if not edges:
        return []

    outgoing: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for s, e in edges:
        outgoing.setdefault(s, []).append(e)

    def polygon_area(poly: list[Point]) -> float:
        """Return the signed area of one polygon loop.

        Args:
            poly (`list[Point]`):
                Polygon vertices in order.

        Returns:
            `float`:
                Signed polygon area computed with the shoelace formula.
        """
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for k in range(len(poly)):
            p1 = poly[k]
            p2 = poly[(k + 1) % len(poly)]
            area += p1.x * p2.y - p2.x * p1.y
        return area / 2.0

    def simplify_collinear(poly: list[Point]) -> list[Point]:
        """Drop redundant vertices that lie on straight axis-aligned segments.

        Args:
            poly (`list[Point]`):
                Polygon vertices in order.

        Returns:
            `list[Point]`:
                Simplified polygon with collinear vertices removed.
        """
        if len(poly) <= 3:
            return poly
        simplified: list[Point] = []
        n = len(poly)
        for k in range(n):
            prev_p = poly[(k - 1) % n]
            curr_p = poly[k]
            next_p = poly[(k + 1) % n]
            same_x = prev_p.x == curr_p.x == next_p.x
            same_y = prev_p.y == curr_p.y == next_p.y
            if same_x or same_y:
                continue
            simplified.append(curr_p)
        return simplified

    polygons: list[list[Point]] = []
    # Trace all closed loops and keep the largest outer one.
    while True:
        starts = [k for k, v in outgoing.items() if v]
        if not starts:
            break

        start = min(starts, key=lambda p: (p[1], p[0]))
        loop: list[tuple[float, float]] = [start]
        current = start
        safety = len(edges) + 5
        for _ in range(safety):
            next_list = outgoing.get(current)
            if not next_list:
                break
            nxt = next_list.pop()
            loop.append(nxt)
            current = nxt
            if current == start:
                break

        if len(loop) >= 4 and loop[-1] == loop[0]:
            points = [Point(x, y) for x, y in loop[:-1]]
            polygons.append(simplify_collinear(points))

    if not polygons:
        return []

    return max(polygons, key=lambda poly: abs(polygon_area(poly)))


def compute_session_orthogonal_polygon(
    session_node_indices: Iterable[int],
    *,
    nodes_per_row: int = NODES_PER_ROW,
    horizontal_spacing: float = HORIZONTAL_SPACING,
    vertical_spacing: float = VERTICAL_SPACING,
    padding_x: float = 0.18,
    padding_y: float = 0.18,
    connector_width_ratio: float = 0.16,
) -> list[Point]:
    """Compute a tight orthogonal polygon around one session path.

    Args:
        session_node_indices (`Iterable[int]`):
            Timeline node indices that belong to one session.
        nodes_per_row (`int`, defaults to `NODES_PER_ROW`):
            Number of nodes shown in each visual row.
        horizontal_spacing (`float`, defaults to `HORIZONTAL_SPACING`):
            Horizontal spacing between adjacent columns.
        vertical_spacing (`float`, defaults to `VERTICAL_SPACING`):
            Vertical spacing between adjacent rows.
        padding_x (`float`, defaults to `0.18`):
            Horizontal padding ratio applied around row rectangles.
        padding_y (`float`, defaults to `0.18`):
            Vertical padding ratio applied around row rectangles.
        connector_width_ratio (`float`, defaults to `0.16`):
            Width ratio used for connector rectangles across rows.

    Returns:
        `list[Point]`:
            Orthogonal polygon vertices suitable for SVG or canvas rendering.
    """
    unique_indices = sorted(set(session_node_indices))
    if not unique_indices:
        raise ValueError("session_node_indices must not be empty")
    if nodes_per_row <= 0:
        raise ValueError("nodes_per_row must be > 0")

    rectangles = _build_session_cover_rectangles(
        unique_indices,
        nodes_per_row=nodes_per_row,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        padding_x=padding_x,
        padding_y=padding_y,
        connector_width_ratio=connector_width_ratio,
    )
    return _rectangles_to_orthogonal_polygon(rectangles)
