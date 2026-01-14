"""
Geometric utility functions for 2D nesting.
"""
from typing import Tuple, List, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import numpy as np


# Tolerance for floating point comparisons
TOLERANCE = 1e-7


def rotate_polygon(polygon: Polygon, rotation: int) -> Polygon:
    """
    Rotate a polygon by multiples of 90 degrees around the origin.

    Args:
        polygon: The polygon to rotate
        rotation: Rotation index (0=0°, 1=90°, 2=180°, 3=270°)

    Returns:
        Rotated polygon
    """
    angle = (rotation % 4) * 90
    if angle == 0:
        return polygon
    return affinity.rotate(polygon, angle, origin=(0, 0))


def translate_polygon(polygon: Polygon, dx: float, dy: float) -> Polygon:
    """
    Translate a polygon by (dx, dy).

    Args:
        polygon: The polygon to translate
        dx: X displacement
        dy: Y displacement

    Returns:
        Translated polygon
    """
    return affinity.translate(polygon, dx, dy)


def scale_polygon(polygon: Polygon, factor: float) -> Polygon:
    """
    Scale a polygon uniformly around the origin.

    Args:
        polygon: The polygon to scale
        factor: Scale factor (negative to reflect)

    Returns:
        Scaled polygon
    """
    return affinity.scale(polygon, xfact=factor, yfact=factor, origin=(0, 0))


def normalize_polygon(polygon: Polygon) -> Polygon:
    """
    Normalize a polygon by translating its bottom-left vertex to origin.

    Args:
        polygon: The polygon to normalize

    Returns:
        Normalized polygon with bottom-left at (0, 0)
    """
    minx, miny, _, _ = polygon.bounds
    return translate_polygon(polygon, -minx, -miny)


def find_bottom_left_vertex(polygon: Polygon) -> Tuple[float, float]:
    """
    Find the bottom-left vertex (minimum y, then minimum x).

    Args:
        polygon: The polygon to search

    Returns:
        (x, y) coordinates of the bottom-left vertex
    """
    coords = list(polygon.exterior.coords)[:-1]  # Exclude closing point
    # Sort by y first, then by x
    bottom_left = min(coords, key=lambda p: (p[1], p[0]))
    return bottom_left


def get_polygon_vertices(polygon: Polygon) -> List[Tuple[float, float]]:
    """
    Get all vertices of a polygon (excluding holes).

    Args:
        polygon: The polygon

    Returns:
        List of (x, y) coordinates
    """
    return list(polygon.exterior.coords)[:-1]  # Exclude closing point


def simplify_polygon(polygon: Polygon, tolerance: float = 0.01) -> Polygon:
    """
    Simplify a polygon by removing nearly collinear points.

    Args:
        polygon: The polygon to simplify
        tolerance: Distance tolerance for simplification

    Returns:
        Simplified polygon
    """
    return polygon.simplify(tolerance, preserve_topology=True)


def remove_collinear_points(polygon: Polygon, tolerance: float = TOLERANCE) -> Polygon:
    """
    Remove collinear points from a polygon.

    Args:
        polygon: The polygon to clean
        tolerance: Angular tolerance

    Returns:
        Polygon with collinear points removed
    """
    coords = list(polygon.exterior.coords)[:-1]  # Exclude closing point
    if len(coords) < 3:
        return polygon

    cleaned = []
    n = len(coords)

    for i in range(n):
        p_prev = coords[(i - 1) % n]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]

        # Check if three points are collinear
        if not _are_collinear(p_prev, p_curr, p_next, tolerance):
            cleaned.append(p_curr)

    if len(cleaned) < 3:
        return polygon

    return Polygon(cleaned)


def _are_collinear(p1: Tuple[float, float], p2: Tuple[float, float],
                   p3: Tuple[float, float], tolerance: float = TOLERANCE) -> bool:
    """
    Check if three points are collinear.

    Uses the cross product: if (p2-p1) x (p3-p1) ≈ 0, points are collinear.
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return abs(cross) < tolerance


def polygon_area(polygon: Polygon) -> float:
    """
    Calculate the area of a polygon (handles holes).

    Args:
        polygon: The polygon

    Returns:
        Absolute area value
    """
    return abs(polygon.area)


def is_valid_polygon(polygon: Polygon) -> bool:
    """
    Check if a polygon is valid (simple and non-degenerate).

    Args:
        polygon: The polygon to check

    Returns:
        True if valid
    """
    return polygon.is_valid and not polygon.is_empty and polygon.area > TOLERANCE


def ensure_ccw(polygon: Polygon) -> Polygon:
    """
    Ensure the polygon exterior is counter-clockwise.

    Args:
        polygon: The polygon

    Returns:
        Polygon with CCW exterior
    """
    if not polygon.exterior.is_ccw:
        return Polygon(list(polygon.exterior.coords)[::-1])
    return polygon


def get_bounding_box(polygon: Polygon) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of a polygon.

    Args:
        polygon: The polygon

    Returns:
        (minx, miny, maxx, maxy)
    """
    return polygon.bounds


def point_in_polygon(polygon: Polygon, x: float, y: float) -> bool:
    """
    Check if a point is inside a polygon.

    Args:
        polygon: The polygon
        x, y: Point coordinates

    Returns:
        True if point is inside (not on boundary)
    """
    point = Point(x, y)
    return polygon.contains(point)


def point_on_boundary(polygon: Polygon, x: float, y: float,
                      tolerance: float = TOLERANCE) -> bool:
    """
    Check if a point is on the polygon boundary.

    Args:
        polygon: The polygon
        x, y: Point coordinates
        tolerance: Distance tolerance

    Returns:
        True if point is on boundary
    """
    point = Point(x, y)
    return polygon.exterior.distance(point) < tolerance


def distance_to_boundary(polygon: Polygon, x: float, y: float) -> float:
    """
    Calculate the distance from a point to the polygon boundary.

    Args:
        polygon: The polygon
        x, y: Point coordinates

    Returns:
        Distance to nearest boundary point
    """
    point = Point(x, y)
    return polygon.exterior.distance(point)


def polygon_offset(polygon: Polygon, distance: float) -> Polygon:
    """
    Offset (buffer) a polygon.
    Positive distance = expand (dilate)
    Negative distance = shrink (erode)

    Args:
        polygon: The polygon to offset
        distance: Offset distance

    Returns:
        Offset polygon
    """
    result = polygon.buffer(distance, join_style=2)  # 2 = mitered
    if result.is_empty or result.area < TOLERANCE:
        return polygon
    # If result is MultiPolygon, take the largest part
    if result.geom_type == 'MultiPolygon':
        result = max(result.geoms, key=lambda p: p.area)
    return result


def polygon_intersection(poly1: Polygon, poly2: Polygon) -> Optional[Polygon]:
    """
    Compute the intersection of two polygons.

    Args:
        poly1, poly2: The polygons

    Returns:
        Intersection polygon or None
    """
    result = poly1.intersection(poly2)
    if result.is_empty:
        return None
    if result.geom_type == 'MultiPolygon':
        result = max(result.geoms, key=lambda p: p.area)
    if result.geom_type != 'Polygon' or result.area < TOLERANCE:
        return None
    return result


def polygon_union(poly1: Polygon, poly2: Polygon) -> Polygon:
    """
    Compute the union of two polygons.

    Args:
        poly1, poly2: The polygons

    Returns:
        Union polygon
    """
    result = poly1.union(poly2)
    if result.geom_type == 'MultiPolygon':
        result = max(result.geoms, key=lambda p: p.area)
    return result


def polygon_difference(poly1: Polygon, poly2: Polygon) -> Optional[Polygon]:
    """
    Compute the difference of two polygons (poly1 - poly2).

    Args:
        poly1, poly2: The polygons

    Returns:
        Difference polygon or None
    """
    result = poly1.difference(poly2)
    if result.is_empty:
        return None
    if result.geom_type == 'MultiPolygon':
        result = max(result.geoms, key=lambda p: p.area)
    if result.geom_type != 'Polygon' or result.area < TOLERANCE:
        return None
    return result
