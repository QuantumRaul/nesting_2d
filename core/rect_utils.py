"""
Rectangle-specific utilities for optimized nesting.

When all shapes are rectangles, we can use simple arithmetic instead of
expensive polygon operations (Minkowski sums, Shapely containment checks).

Key optimizations:
- NFP of two rectangles is always a rectangle: O(1) vs O(n*m)
- Penetration depth is simple AABB overlap: O(1) vs O(n)
- Candidate points are just corners: O(n) vs O(n^2)
"""
from typing import Tuple, List, Optional, Set
from shapely.geometry import Polygon

# Tolerance for floating-point comparisons
TOLERANCE = 1e-7


def is_rectangle(polygon: Polygon, tolerance: float = 1e-6) -> bool:
    """
    Check if a polygon is an axis-aligned rectangle.

    Args:
        polygon: Shapely Polygon to check
        tolerance: Tolerance for floating-point comparisons

    Returns:
        True if polygon is an axis-aligned rectangle
    """
    if polygon is None or polygon.is_empty:
        return False

    # Get exterior coordinates (excluding the closing point)
    coords = list(polygon.exterior.coords)[:-1]

    # Must have exactly 4 vertices
    if len(coords) != 4:
        return False

    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds

    # Check if all vertices are at bounding box corners
    corners = {
        (minx, miny), (minx, maxy),
        (maxx, miny), (maxx, maxy)
    }

    for x, y in coords:
        found = False
        for cx, cy in corners:
            if abs(x - cx) < tolerance and abs(y - cy) < tolerance:
                found = True
                break
        if not found:
            return False

    return True


def polygon_to_rect_dims(polygon: Polygon) -> Tuple[float, float]:
    """
    Extract (width, height) from a rectangle polygon.

    Assumes the polygon is already normalized (bottom-left at origin).

    Args:
        polygon: Shapely Polygon (should be a rectangle)

    Returns:
        (width, height) tuple
    """
    minx, miny, maxx, maxy = polygon.bounds
    return (maxx - minx, maxy - miny)


def compute_rect_nfp(w_a: float, h_a: float,
                     w_b: float, h_b: float) -> Tuple[float, float, float, float]:
    """
    Compute NFP for two axis-aligned rectangles.

    For rectangles A (stationary) and B (orbiting), the NFP is simply
    another rectangle with dimensions (w_a + w_b, h_a + h_b).

    The NFP is positioned such that when B's reference point (bottom-left)
    is at the NFP origin, B is touching A's bottom-left corner.

    Args:
        w_a, h_a: Width and height of stationary rectangle A
        w_b, h_b: Width and height of orbiting rectangle B

    Returns:
        (nfp_x, nfp_y, nfp_width, nfp_height) - NFP bounds
    """
    nfp_width = w_a + w_b
    nfp_height = h_a + h_b

    # NFP origin: when B's reference is here, B is left and below A
    nfp_x = -w_b
    nfp_y = -h_b

    return (nfp_x, nfp_y, nfp_width, nfp_height)


def compute_rect_pd(x1: float, y1: float, w1: float, h1: float,
                    x2: float, y2: float, w2: float, h2: float) -> float:
    """
    Compute penetration depth between two axis-aligned rectangles.

    PD is the minimum distance one rectangle must move to eliminate overlap.
    Returns 0 if rectangles don't overlap.

    Args:
        x1, y1, w1, h1: Bounds of rectangle 1 (x, y, width, height)
        x2, y2, w2, h2: Bounds of rectangle 2 (x, y, width, height)

    Returns:
        Penetration depth (0 if no overlap)
    """
    # Compute overlap in each dimension
    overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
    overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)

    # No overlap if either dimension has no overlap
    if overlap_x <= TOLERANCE or overlap_y <= TOLERANCE:
        return 0.0

    # PD is the minimum separation needed
    return min(overlap_x, overlap_y)


def compute_rect_pd_via_nfp(nfp_x: float, nfp_y: float,
                            nfp_w: float, nfp_h: float,
                            rel_x: float, rel_y: float) -> float:
    """
    Compute penetration depth using NFP-based approach.

    If the relative position (rel_x, rel_y) is inside the NFP,
    the rectangles overlap. PD is the distance to the nearest NFP edge.

    Args:
        nfp_x, nfp_y, nfp_w, nfp_h: NFP bounds
        rel_x, rel_y: Relative position of B's reference to A's reference

    Returns:
        Penetration depth (0 if no overlap)
    """
    # Check if point is inside NFP (strictly inside = overlap)
    if not (nfp_x < rel_x < nfp_x + nfp_w and
            nfp_y < rel_y < nfp_y + nfp_h):
        return 0.0

    # Distance to each NFP edge
    dist_left = rel_x - nfp_x
    dist_right = (nfp_x + nfp_w) - rel_x
    dist_bottom = rel_y - nfp_y
    dist_top = (nfp_y + nfp_h) - rel_y

    return min(dist_left, dist_right, dist_bottom, dist_top)


def rect_ifr(sheet_w: float, sheet_h: float,
             part_w: float, part_h: float) -> Tuple[float, float, float, float]:
    """
    Compute Inner-Fit Rectangle for a rectangle part in a rectangular sheet.

    The IFR is the region where the part's reference point (bottom-left)
    can be placed such that the part stays entirely within the sheet.

    Args:
        sheet_w, sheet_h: Sheet dimensions
        part_w, part_h: Part dimensions

    Returns:
        (ifr_x, ifr_y, ifr_width, ifr_height) - IFR bounds
        Returns (0, 0, 0, 0) if part doesn't fit
    """
    ifr_w = sheet_w - part_w
    ifr_h = sheet_h - part_h

    if ifr_w < 0 or ifr_h < 0:
        return (0.0, 0.0, 0.0, 0.0)

    return (0.0, 0.0, ifr_w, ifr_h)


def is_collision_free_rect(x: float, y: float, w: float, h: float,
                           placed_rects: List[Tuple[float, float, float, float]]) -> bool:
    """
    Check if placing a rectangle at (x, y) causes any collision.

    Args:
        x, y: Proposed position (bottom-left)
        w, h: Rectangle dimensions
        placed_rects: List of (x, y, width, height) for placed rectangles

    Returns:
        True if no collision
    """
    for px, py, pw, ph in placed_rects:
        # Check AABB overlap
        overlap_x = min(x + w, px + pw) - max(x, px)
        overlap_y = min(y + h, py + ph) - max(y, py)

        if overlap_x > TOLERANCE and overlap_y > TOLERANCE:
            return False

    return True


def get_rect_candidate_points(
    ifr: Tuple[float, float, float, float],
    placed_rects: List[Tuple[float, float, float, float]],
    new_dims: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Generate candidate placement points for a rectangle.

    For rectangles, optimal placements are typically at corners:
    - IFR corners
    - Points adjacent to placed rectangles (touching edges)

    Args:
        ifr: IFR bounds (x, y, width, height)
        placed_rects: List of (x, y, width, height) for placed rectangles
        new_dims: (width, height) of rectangle to place

    Returns:
        List of (x, y) candidate positions, sorted bottom-left first
    """
    ifr_x, ifr_y, ifr_w, ifr_h = ifr
    new_w, new_h = new_dims

    if ifr_w < 0 or ifr_h < 0:
        return []

    points: Set[Tuple[float, float]] = set()

    # IFR corners
    points.add((ifr_x, ifr_y))
    if ifr_w > 0:
        points.add((ifr_x + ifr_w, ifr_y))
    if ifr_h > 0:
        points.add((ifr_x, ifr_y + ifr_h))
    if ifr_w > 0 and ifr_h > 0:
        points.add((ifr_x + ifr_w, ifr_y + ifr_h))

    # Points at corners of placed rectangles
    for rx, ry, rw, rh in placed_rects:
        # Right side of placed rectangle
        points.add((rx + rw, ry))
        points.add((rx + rw, ry + rh - new_h))

        # Top side of placed rectangle
        points.add((rx, ry + rh))
        points.add((rx + rw - new_w, ry + rh))

        # Left side (if there's room)
        if rx >= new_w:
            points.add((rx - new_w, ry))
            points.add((rx - new_w, ry + rh - new_h))

        # Bottom side (if there's room)
        if ry >= new_h:
            points.add((rx, ry - new_h))
            points.add((rx + rw - new_w, ry - new_h))

        # Additional corner combinations
        points.add((rx + rw, ry + rh))
        points.add((rx, ry))

    # Filter to valid IFR range
    valid_points = []
    for px, py in points:
        if (ifr_x - TOLERANCE <= px <= ifr_x + ifr_w + TOLERANCE and
            ifr_y - TOLERANCE <= py <= ifr_y + ifr_h + TOLERANCE):
            # Clamp to IFR bounds
            px = max(ifr_x, min(px, ifr_x + ifr_w))
            py = max(ifr_y, min(py, ifr_y + ifr_h))
            valid_points.append((px, py))

    # Sort by x then y (bottom-left first for greedy placement)
    valid_points.sort(key=lambda p: (p[0], p[1]))

    # Remove duplicates after clamping
    unique_points = []
    seen = set()
    for p in valid_points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)

    return unique_points


def get_rect_arrangement_points(
    ifr: Tuple[float, float, float, float],
    placed_rects: List[Tuple[float, float, float, float]],
    new_dims: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Generate arrangement points for overlap minimization.

    Similar to get_rect_candidate_points but includes more corner
    combinations for thorough search during optimization.

    Args:
        ifr: IFR bounds (x, y, width, height)
        placed_rects: List of (x, y, width, height) for placed rectangles
        new_dims: (width, height) of rectangle to place

    Returns:
        List of (x, y) candidate positions
    """
    ifr_x, ifr_y, ifr_w, ifr_h = ifr
    new_w, new_h = new_dims

    if ifr_w < 0 or ifr_h < 0:
        return []

    points: Set[Tuple[float, float]] = set()

    # IFR corners
    points.add((ifr_x, ifr_y))
    points.add((ifr_x + ifr_w, ifr_y))
    points.add((ifr_x, ifr_y + ifr_h))
    points.add((ifr_x + ifr_w, ifr_y + ifr_h))

    # For each placed rectangle, generate multiple candidate points
    for rx, ry, rw, rh in placed_rects:
        # All corners of the placed rectangle
        corners = [
            (rx, ry),           # bottom-left
            (rx + rw, ry),      # bottom-right
            (rx, ry + rh),      # top-left
            (rx + rw, ry + rh)  # top-right
        ]

        for cx, cy in corners:
            # Position new rect with various alignments relative to corner
            points.add((cx, cy))
            points.add((cx - new_w, cy))
            points.add((cx, cy - new_h))
            points.add((cx - new_w, cy - new_h))

        # Edge midpoints
        points.add((rx + rw / 2, ry))
        points.add((rx + rw / 2, ry + rh))
        points.add((rx, ry + rh / 2))
        points.add((rx + rw, ry + rh / 2))

    # Filter to valid IFR range
    valid_points = []
    for px, py in points:
        if (ifr_x - TOLERANCE <= px <= ifr_x + ifr_w + TOLERANCE and
            ifr_y - TOLERANCE <= py <= ifr_y + ifr_h + TOLERANCE):
            px = max(ifr_x, min(px, ifr_x + ifr_w))
            py = max(ifr_y, min(py, ifr_y + ifr_h))
            valid_points.append((px, py))

    # Remove duplicates
    unique_points = []
    seen = set()
    for p in valid_points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)

    # Shuffle for randomness in optimization
    import random
    random.shuffle(unique_points)

    return unique_points


def get_rotated_dims(w: float, h: float, rotation: int) -> Tuple[float, float]:
    """
    Get rectangle dimensions after rotation.

    Args:
        w, h: Original width and height
        rotation: Rotation index (0=0°, 1=90°, 2=180°, 3=270°)

    Returns:
        (width, height) after rotation
    """
    if rotation % 2 == 0:
        return (w, h)
    else:
        return (h, w)
